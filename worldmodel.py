from collections import deque
import collections
from statistics import mean, stdev
import random
import curses
import argparse
import time
import pathlib
import yaml
import re
import types
import importlib

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
import torch.backends.cudnn
import torch.cuda
import numpy as np
import warnings
import wandb

from models.models import Policy
from wm2.viz import Viz, DummyCurses
from wm2.data.datasets import Buffer, SARDataset, SARNextDataset, SimpleRewardDataset, DummyBuffer, \
    SubsetSequenceBuffer
from wm2.utils import Pbar
from wm2.data.utils import pad_collate_2
import wm2.utils
import wm2.env.wrappers


def mse_loss(trajectories, predicted_state):
    loss = ((trajectories.next_state.to(args.device) - predicted_state) ** 2) * trajectories.pad.to(
        args.device)
    return loss.mean()


def log_prob_loss_simple(trajectories, predicted_state):
    return - predicted_state.log_prob(trajectories.next_state.to(args.device)).mean()


def log_prob_loss_entropy(trajectories, predicted_state):
    prior = Normal(predicted_state.loc[0:-1], predicted_state.scale[0:-1])
    posterior = Normal(predicted_state.loc[1:], predicted_state.scale[1:])
    # div = kl_divergence(prior, posterior).mean()
    log_p = predicted_state.log_prob(trajectories.next_state.to(args.device)).mean()
    entropy = posterior.entropy().mean()
    return - log_p - entropy


class Identity:
    def __call__(self, input):
        return input


class EnvObserver:
    def __init__(self):
        pass

    def reset(self):
        """ called before environment reset"""
        pass

    def step(self, state, action, reward, done, info, action_dist, sampled_action):
        """ called each environment step """
        pass

    def end(self):
        """ called when episode ends """
        pass


class BufferWriter(EnvObserver):
    def __init__(self, buffer, state_prepro, reward_prepro):
        super().__init__()
        self.b = buffer
        self.state_prepro = state_prepro
        self.reward_prepro = reward_prepro
        self.current_trajectory = 0

    def reset(self):
        self.current_trajectory = self.b.next_traj()

    def append(self, state, action, reward, done, info, action_dist, sampled_action):
        self.current_trajectory.append(self.state_prepro(state),
                                       action,
                                       self.reward_prepro(reward),
                                       done,
                                       info)


class EnvRunner:
    def __init__(self, args, env, action_pipeline, seed=None):
        self.args = args
        self.env = env
        if seed is not None:
            env.seed(seed)
        self.observers = {}
        self.action_pipeline = action_pipeline

    def attach(self, name, observer):
        self.observers[name] = observer

    def detach(self, name):
        del self.observers[name]

    def observer_reset(self):
        for name, observer in self.observers.items():
            observer.reset()

    def observe_step(self, state, action, reward, done, info, action_dist, sampled_action):
        for name, observer in self.observers.items():
            observer.append(state, action, reward, done, info, action_dist, sampled_action)

    def observer_episode_end(self):
        for name, observer in self.observers.items():
            observer.end()

    def render(self, render, delay):
        if render:
            self.env.render()
            time.sleep(delay)

    def episode(self, args, policy, render=False, delay=0.01):
        with torch.no_grad():
            self.observer_reset()
            state, reward, done, info = self.env.reset(), 0.0, False, {}
            action_dist, sampled_action, action = self.action_pipeline(args, state, policy)
            self.observe_step(state, action, reward, done, info, action_dist, sampled_action)
            self.render(render, delay)
            while not done:
                state, reward, done, info = self.env.step(action)
                action_dist, sampled_action, action = self.action_pipeline(args, state, policy)
                self.observe_step(state, action, reward, done, info, action_dist, sampled_action)
                self.render(render, delay)

            self.observer_episode_end()


def gather_experience(buff, episode, env, connector, policy, eps=0.0, expln_noise=0.0, render=True, seed=None,
                      delay=0.01):
    with torch.no_grad():

        def get_action(state, reward, done):
            t_state = connector.policy_prepro(state, args.device).unsqueeze(0)

            if random.random() >= eps:
                action_dist = policy(t_state)
                action = action_dist.rsample()
                action = Normal(action, expln_noise).sample()
                action = action.clamp(min=-1.0, max=1.0)
            else:
                action_dist = eps_policy(t_state)
                action = action_dist.rsample()
            action = connector.action_prepro(action)

            buff.append(episode,
                        connector.buffer_prepro(state),
                        action,
                        connector.reward_prepro(reward),
                        done,
                        None)
            if render:
                env.render()
                time.sleep(delay)
                # print(reward, state[-2], state[-3])
            return action, action_dist

        # gather new experience
        episode_reward = []
        episode_entropy = []
        eps_policy = connector.make_random_policy(env)

        if seed is not None:
            env.seed(seed)

        state, reward, done = env.reset(), 0.0, False

        action, action_dist = get_action(state, reward, done)
        episode_entropy += [action_dist.entropy().mean().item()]
        episode_reward += [reward]

        while not done:
            state, reward, done, info = env.step(action)
            action, action_dist = get_action(state, reward, done)

            episode_entropy += [action_dist.entropy().mean().item()]
            episode_reward += [reward]

    return buff, episode_reward, episode_entropy


def gather_seed_episodes(env, connector, random_policy, buff, seed_episodes):
    for episode in range(seed_episodes):
        gather_experience(buff, episode, env, connector, random_policy, render=False)
    return buff


def make_connector(args):
    module_name, connector_class_name = args.connector.split(':')
    connector_module = importlib.import_module(module_name)
    connector = getattr(connector_module, connector_class_name)
    connector = connector(**vars(args))
    return connector


def determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """ monitoring """
    scr = DummyCurses()
    recent_reward = deque(maxlen=20)
    wandb.gym.monitor()
    imagine_log_cooldown = wm2.utils.Cooldown(secs=args.viz_imagine_log_cooldown)
    viz_cooldown = wm2.utils.Cooldown(secs=args.viz_cooldown)
    render_cooldown = wm2.utils.Cooldown(secs=args.viz_render_cooldown)
    episode_refresh = wm2.utils.Cooldown(secs=args.viz_episode_refresh)
    viz = Viz(args=args, window_title=f'{wandb.run.project} {wandb.run.id}')

    """ experience buffers """
    train_buff = Buffer(p_cont_algo=args.pcont_algo, terminal_repeats=args.pcont_terminal_repeats)
    test_buff = Buffer(p_cont_algo=args.pcont_algo, terminal_repeats=args.pcont_terminal_repeats)
    train_episode, test_episode = args.seed_episodes, args.seed_episodes
    dummy_buffer = DummyBuffer()

    """ connect environment """
    connector = make_connector(args)
    env = connector.make_env(args)
    action_pipeline = connector.make_action_pipeline()
    train_runner = EnvRunner(args, env, action_pipeline)
    train_runner.attach('train_buffer', BufferWriter(train_buff, connector.buffer_prepro, connector.reward_prepro))
    test_runner = EnvRunner(args, env, action_pipeline)
    test_runner.attach('train_buffer', BufferWriter(test_buff, connector.buffer_prepro, connector.reward_prepro))
    random_policy = connector.make_random_policy(env)
    env_viz = connector.make_viz(args)

    train_runner.episode(args, random_policy, True)
    test_runner.episode(args, random_policy, True)

    # train_buff = gather_seed_episodes(env, connector, random_policy, train_buff, args.seed_episodes)
    # test_buff = gather_seed_episodes(env, connector, random_policy, test_buff, args.seed_episodes)

    """ policy model """
    policy, policy_optim = connector.make_policy(args)

    """ value model"""
    value, value_optim = connector.make_value(args)

    """ probability of continuing """
    pcont, pcont_optim = connector.make_pcont_model(args)

    """ dynamics function or model"""
    T = connector.make_transition_model(args).to(args.device)
    T_optim = Adam(T.parameters(), lr=args.dynamics_lr) if args.dynamics_learnable else None

    """ reward function or model"""
    R = connector.make_reward_model(args).to(args.device)
    R_optim = Adam(R.parameters(), lr=args.reward_lr) if args.reward_learnable else None

    "save and restore helpers"
    R_saver = wm2.utils.SaveLoad('reward')
    policy_saver = wm2.utils.SaveLoad('policy')
    value_saver = wm2.utils.SaveLoad('value')
    pcont_saver = wm2.utils.SaveLoad('pcont')

    converged = False
    scr.clear()
    best_stats = {'mean': 0}
    iteration = 0

    while not converged:
        iteration += 1

        for c in range(args.collect_interval):

            sample_train_buff = SubsetSequenceBuffer(train_buff, args.batch_size, args.horizon + 1)
            sample_test_buff = SubsetSequenceBuffer(test_buff, args.batch_size, args.horizon + 1)

            scr.update_progressbar(c)
            scr.update_slot('wandb', f'{wandb.run.name} {wandb.run.project} {wandb.run.id}')
            scr.update_slot('buffer_stats',
                            f'train_buff size {len(sample_train_buff)} rejects {sample_train_buff.rejects}')

            def train_reward():
                """ train the reward model
                note: since rewards are sparse, this is the most difficult to model
                therefore hand-coding your reward function, if possible, is much superior """

                # train_weights, test_weights = weights(sample_train_buff, log=True), weights(sample_test_buff)
                # train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
                # test_sampler = WeightedRandomSampler(test_weights, len(test_weights), replacement=True)
                # train = DataLoader(train, batch_size=40*50, sampler=train_sampler)
                # test = DataLoader(test, batch_size=40*50, sampler=test_sampler)

                train, test = SimpleRewardDataset(sample_train_buff), SimpleRewardDataset(sample_test_buff)
                train = DataLoader(train, batch_size=args.batch_size * 50, shuffle=True)
                test = DataLoader(test, batch_size=args.batch_size * 50, shuffle=True)

                R.train()

                for state, reward in train:
                    R_optim.zero_grad()
                    predicted_reward = R(state.to(args.device))
                    loss = ((reward.to(args.device) - predicted_reward) ** 2).mean()
                    loss.backward()
                    R_optim.step()
                    scr.update_slot('reward_train', f'Reward train loss {loss.item()}')
                    wandb.log({'reward_train': loss.item()})
                    R_saver.checkpoint(R, R_optim)

                R.eval()

                for state, reward in test:
                    predicted_reward = R(state.to(args.device))
                    loss = ((reward.to(args.device) - predicted_reward) ** 2).mean()
                    scr.update_slot('reward_test', f'Reward test loss  {loss.item()}')
                    wandb.log({'reward_test': loss.item()})

            def train_pcont():

                """ probability of continuing: what is the chance the episode will continue in the next step """
                """ note that this is a proxy value function, it's estimated using a bellman-type update """
                train, test = SARDataset(sample_train_buff, mask_f=None), SARDataset(sample_test_buff, mask_f=None)
                train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
                test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

                " regress pcont against the done flag "
                pcont.train()
                for trajectories in train:
                    state = trajectories.state[:-1].to(args.device)
                    done = trajectories.done[:-1].to(args.device)
                    next_state = trajectories.state[1:].to(args.device)

                    pcont_optim.zero_grad()
                    target = (1.0 - done) * pcont(next_state).detach()
                    pred = pcont(state)
                    loss = ((target - pred) ** 2).mean()
                    loss.backward()
                    pcont_optim.step()

                pcont.eval()
                #
                # pcont.train()
                # for trajectories in train:
                #     pcont_optim.zero_grad()
                #     predicted_pcont = pcont(trajectories.state.to(args.device))
                #     loss = ((predicted_pcont - trajectories.pcont.to(args.device)) ** 2).mean()
                #     loss.backward()
                #     pcont_optim.step()
                #     scr.update_slot('pcont_train', f'pcont train loss  {loss.item()}')
                # pcont.eval()

                # for trajectories in test:
                #     predicted_pcont = pcont(trajectories.state.to(args.device))
                #     loss = ((predicted_pcont - trajectories.pcont.to(args.device)) ** 2).mean()
                #     scr.update_slot('pcont_test', f'pcont test loss  {loss.item()}')

            def train_behavior():

                # logging
                update_text = imagine_log_cooldown()
                value_sample = deque(maxlen=100)

                # Behaviour learning
                train = ConcatDataset([SARDataset(sample_train_buff)])
                train = DataLoader(train, batch_size=args.batch_size * 40, collate_fn=pad_collate_2, shuffle=True)

                for trajectory in train:
                    trajectory.state = trajectory.state.to(args.device)
                    trajectory.action = trajectory.action.to(args.device)

                    prediction = T.imagine(args, trajectory, policy)

                    rstack = R(prediction)
                    vstack = value(prediction)
                    pcontstack = pcont(prediction)

                    vstack = vstack * pcontstack
                    rstack = rstack * pcontstack
                    H, L, N, S = rstack.shape

                    """ construct matrix in form
                    
                    R0 V1  0  0
                    R0 R1 V2  0
                    R0 R1 R2 V3
                    
                    Where R0, R1, R2 are predicted rewards at timesteps 0, 1, 2
                    and V1, V2, V3 are the predicted future values of states at time 1, 2, 3
                    """

                    # create a H * H matrix filled with rewards (using rstack preserves computation graph)
                    rstack = rstack.repeat(H, 1, 1, 1).reshape(H, H, L, N, S)

                    # replace the upper triangle with zeros
                    r, c = torch.triu_indices(H, H)
                    rstack[r, c] = 0.0

                    # replace diagonal rewards with values
                    rstack[torch.arange(H), torch.arange(H)] = vstack[torch.arange(H)]

                    # clip the top row
                    rstack = rstack[1:, :]

                    # occasionally dump table to screen for analysis
                    if update_text:
                        prob_cont = pcontstack[:, 0, 0, 0].detach().cpu().numpy()
                        scr.update_table('pcont', prob_cont)
                        rewards = rstack[-1:, :, 0, 0, 0].detach().cpu().numpy()
                        scr.update_table('rewards', rewards)
                        v = vstack[:, 0, 0, 0].unsqueeze(0).detach().cpu().numpy()
                        scr.update_table('values', v)
                        # imagined_trajectory = torch.stack(imagine)[:, 0, 0, :].detach().cpu().numpy().T
                        # scr.update_table('imagined trajectory', imagined_trajectory)

                    """ reduce the above matrix to values using the formula from the paper in 2 steps
                    first, compute VN for each k by applying discounts to each time-step and compute the expected value
                    """

                    # compute and apply the discount (alternatively can estimate the discount using a probability to terminate function)
                    n = torch.linspace(0.0, H - 1, H, device=args.device)
                    discount = torch.full_like(n, args.discount, device=args.device).pow(n).view(1, -1, 1, 1, 1)
                    rstack = rstack * discount

                    # compute the expectation
                    steps = torch.linspace(2, H, H - 1, device=args.device).view(-1, 1, 1, 1)
                    VNK = rstack.sum(dim=1) / steps
                    if update_text:
                        scr.update_table('VNK', VNK[:, 0, 0, 0].unsqueeze(0).detach().cpu().numpy())

                    """ now we are left with a single column matrix in form
                    VN(k=1)
                    VN(k=2)
                    VN(k=3)
                    
                    combine these into a single value with the V lambda equation
                    
                    VL = (1-lambda) * lambda ^ 0  * VN(k=1) + (1 - lambda) * lambda ^ 1 VN(k=2) + lambda ^ 2 * VN(k=3)
                    
                    Note the lambda terms should sum to 1, or you're doing it wrong
                    """

                    lam = torch.full((VNK.size(0),), args.lam, device=args.device).pow(
                        torch.arange(VNK.size(0), device=args.device)).view(-1, 1, 1, 1)
                    lam[0:-1] = lam[0:-1] * (1 - args.lam)
                    VL = (VNK * lam).sum(0)
                    if update_text:
                        scr.update_table('VL', VL[:, 0, 0].detach().cpu().numpy())

                    "backprop loss through policy"
                    policy_optim.zero_grad()
                    policy_loss = -VL.mean()
                    policy_loss.backward()
                    if args.policy_clip > 0.0:
                        clip_grad_norm_(parameters=policy.parameters(), max_norm=args.policy_clip)
                    policy_optim.step()

                    "housekeeping"
                    viz.sample_grad_norm(policy, sample=0.05)
                    scr.update_slot('policy_loss', f'Policy loss  {policy_loss.item()}')
                    wandb.log({'policy_loss': policy_loss.item()})
                    policy_saver.checkpoint(policy, policy_optim)

                    " regress value against tau ie: the initial estimated value... "
                    value.train()
                    value_optim.zero_grad()
                    VN = VL.detach()
                    value_sample.append(VN.clone().detach().cpu().flatten().numpy())
                    values = value(trajectory.state)
                    value_loss = ((VN - values) ** 2).mean() / 2
                    value_loss.backward()
                    if args.value_clip > 0.0:
                        clip_grad_norm_(parameters=value.parameters(), max_norm=args.value_clip)
                    value_optim.step()
                    value.eval()

                    "housekeeping"
                    scr.update_slot('value_loss', f'Value loss  {value_loss.item()}')
                    wandb.log({'value_loss': value_loss.item()})
                    value_saver.checkpoint(value, value_optim)
                    if value_saver.is_best(value_loss):
                        value_saver.save(value, 'best')
                    if update_text:
                        viz.update_sampled_values(value_sample)

            if args.dynamics_learnable:
                T.learn(args, sample_train_buff, T_optim)
            if args.reward_learnable:
                train_reward()
            if not args.pcont_fixed_length:
                train_pcont()

            train_behavior()

        "run the policy on the environment and collect experience"
        # train_buff, reward, entropy = gather_experience(train_buff, train_episode, env, connector, policy,
        #                                                 eps=0.0, seed=args.seed,
        #                                                 render=render_cooldown(), expln_noise=args.exploration_noise)
        # if episode_refresh():
        #     viz.update_trajectory_plots(value, R, pcont, T, train_buff, train_episode, entropy)
        train_runner.episode(args, policy, render=True)

        train_episode += 1

        # wandb.log({'reward': reward})
        # viz.update_rewards(reward)
        # recent_reward.append(sum(reward))
        # scr.update_slot('eps', f'exploration_noise: {args.exploration_noise} '
        #                        f'forward_slope: {args.forward_slope} '
        #                        f'pcont_algo {args.pcont_algo} pcont_terminal_repeats {args.pcont_terminal_repeats}')
        # rr = ''
        # for reward in recent_reward:
        #     rr += f' {reward:.5f},'
        # scr.update_slot('recent_rewards', 'Recent rewards: ' + rr)
        # scr.update_slot('beat_ave_reward', f'Best {best_stats}')
        #
        # "check if the policy is worth saving"
        # if reward > best_stats['mean']:
        #     sampled_rewards = []
        #
        #     for _ in range(args.best_policy_sample_n):
        #         dummy_buffer, reward, entropy = gather_experience(dummy_buffer, train_episode, env, connector, policy,
        #                                                           eps=0.0,
        #                                                           expln_noise=0.0, seed=args.seed)
        #         sampled_rewards.append(sum(reward))
        #     if mean(sampled_rewards) > best_stats['mean']:
        #         best_stats['mean'] = mean(sampled_rewards)
        #         best_stats['std'] = stdev(sampled_rewards)
        #         best_stats['max'] = max(sampled_rewards)
        #         best_stats['min'] = min(sampled_rewards)
        #         best_stats['itr'] = iteration
        #         best_stats['n'] = args.best_policy_sample_n
        #         best_stats['expl'] = args.exploration_noise
        #         best_stats['fw_s'] = args.forward_slope
        #
        #         policy_saver.save(policy, 'best', **best_stats)
        #
        # test_buff, reward, entropy = gather_experience(test_buff, test_episode, env, connector, policy,
        #                                                eps=0.0,
        #                                                render=False, expln_noise=args.exploration_noise, seed=args.seed)
        # test_episode += 1

        if viz_cooldown():
            test_runner.episode(args, policy, render=True)
            viz.plot_rewards_histogram(train_buff, R)
            # viz.update_dynamics(test_buff, T)
            viz.update_pcont(test_buff, pcont)
            viz.update_value(test_buff, value)
            env_viz.update(args, test_buff, policy, R, value, T, pcont)

        converged = False


def demo(dir, env, connector, n=100):
    args.state_dims = env.observation_space.shape[0]
    args.action_dims = env.action_space.shape[0]
    args.action_min = -1.0
    args.action_max = 1.0

    dummy_buffer = DummyBuffer()

    # policy model
    policy = Policy(layers=[args.state_dims, *args.policy_hidden_dims, args.action_dims], min=args.action_min,
                    max=args.action_max).to(args.device)

    iterations = 0

    while iterations < n:
        load_dict = wm2.utils.SaveLoad.load(dir, 'policy', 'best')
        msg = ''
        for arg, value in load_dict.items():
            if arg != 'model':
                msg += f'{arg}: {value} '
        print(msg)
        policy.load_state_dict(load_dict['model'])
        train_buff, reward, entropy = gather_experience(dummy_buffer, 0, env, connector, policy,
                                                        eps=0.0,
                                                        render=True, delay=1.0 / args.fps)
        iterations += 1


def get_args(defaults):
    parser = argparse.ArgumentParser()
    for argument, value in defaults.items():
        if argument == 'seed':
            parser.add_argument('--' + argument, type=int, required=False, default=None)
        elif argument == 'config':
            parser.add_argument('--' + argument, type=str, required=True, default=None)
        else:
            parser.add_argument('--' + argument, type=type(value), required=False, default=None)
    command_line = parser.parse_args()
    """ 
    required due to https://github.com/yaml/pyyaml/issues/173
    pyyaml does not correctly parse scientific notation 
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    def flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    yaml_conf = {}
    if command_line.config is not None:
        with pathlib.Path(command_line.config).open() as f:
            yaml_conf = yaml.load(f, Loader=loader)
            yaml_conf = flatten(yaml_conf)
    args = {}
    """ precedence is command line, config file, default """
    for key, value in vars(command_line).items():
        if value is not None:
            args[key] = vars(command_line)[key]
        elif key in yaml_conf:
            args[key] = yaml_conf[key]
        else:
            args[key] = defaults[key]
    args = types.SimpleNamespace(**args)
    return args


defaults = {
    'name': 'default_run',
    'env': 'HalfCheetahPyBulletEnv-v0',
    'connector': 'wm2.env.connector:EnvConnector',
    'seed_episodes': 10,
    'collect_interval': 10,
    'batch_size': 40,
    'device': 'cuda:0',
    'horizon': 15,
    'discount': 0.99,
    'lam': 0.95,
    'exploration_noise': 0.25,

    'dynamics_learnable': True,
    'dynamics_lr': 1e-4,
    'dynamics_layers': 1,
    'dynamics_hidden_dim': 300,
    'dynamics_reg': 0.5,
    'dynamics_dropout': 0.0,
    'dynamics_ode_method': 'rk4',

    'pcont_fixed_length': False,
    'pcont_lr': 1e-4,
    'pcont_hidden_dims': [48, 48],
    'pcont_nonlin': 'nn.ELU',
    'pcont_algo': 'invexp',
    'pcont_terminal_repeats': 3,

    'value_lr': 2e-5,
    'value_clip': 0.0,
    'value_hidden_dims': [300, 300],
    'value_nonlin': 'nn.ELU',

    'policy_lr': 2e-5,
    'policy_clip': 0.0,
    'policy_hidden_dims': [48, 48],
    'policy_nonlin': 'nn.ELU',

    'reward_learnable': True,
    'reward_lr': 1e-4,
    'reward_hidden_dims': [300, 300],
    'reward_nonlin': 'nn.ELU',

    'forward_slope': 28,

    'viz_cooldown': 240,
    'viz_imagine_log_cooldown': 30,
    'viz_render_cooldown': 240,
    'viz_episode_refresh': 60,

    'best_policy_sample_n': 10,
    'demo': 'off',
    'seed': None,
    'fps': 24,
    'config': None
}

if __name__ == '__main__':

    args = get_args(defaults)

    if args.seed is not None:
        determinism(args.seed)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        if args.demo == 'off':
            wandb.init(config=vars(args))
            curses.wrapper(main(args))
        else:
            wandb_run_dir = str(next(pathlib.Path().glob(f'wandb/*{args.demo}')))
            connector = make_connector(args)
            env = connector.make_env()
            env.render()
            demo(wandb_run_dir, env, connector)
