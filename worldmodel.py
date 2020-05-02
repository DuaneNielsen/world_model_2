from collections import deque
from statistics import mean
from random import random
import curses
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.distributions import Normal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.nn.utils import clip_grad_norm_
import numpy as np
import wandb
import gym
import gym.wrappers

import matplotlib.pyplot as plt

from distributions import ScaledTanhTransformedGaussian
from viz import Curses
from wm2.data.datasets import Buffer, SARDataset, SARNextSubSequenceDataset, SimpleRewardDataset, DummyBuffer
from wm2.utils import Pbar
from data.utils import pad_collate_2
import wm2.utils
from wm2.env.LunarLander_v3 import LunarLanderContinuous, LunarLanderConnector


class MLP(nn.Module):
    def __init__(self, layers, nonlin=None):
        super().__init__()
        in_dims = layers[0]
        net = []
        # the use of eval here is a security bug
        nonlin = nn.ELU if nonlin is None else eval(nonlin)
        for hidden in layers[1:-1]:
            net += [nn.Linear(in_dims, hidden)]
            net += [nonlin()]
            in_dims = hidden
        net += [nn.Linear(in_dims, layers[-1], bias=False)]

        self.mlp = nn.Sequential(*net)

    def forward(self, inp):
        return self.mlp(inp)


class Mixture(nn.Module):
    def __init__(self, state_dims, hidden_dims, n_gaussians=12, nonlin=None):
        super().__init__()
        self.hidden_net = MLP([state_dims, *hidden_dims], nonlin)
        n_hidden = hidden_dims[-1]
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

    def forward(self, inp):
        last_dim = len(inp.shape) - 1
        hidden = torch.tanh(self.hidden_net(inp))
        pi = torch.softmax(self.z_pi(hidden), dim=last_dim)
        mu = self.z_mu(hidden)
        sigma = torch.exp(self.z_sigma(hidden))
        mix = Categorical(probs=pi)
        comp = Normal(mu, sigma)
        gmm = MixtureSameFamily(mix, comp)
        return gmm


class HandcraftedPrior(nn.Module):
    def __init__(self, state_dims, hidden_dims, nonlin=None):
        super().__init__()
        self.crash_detector = MLP([state_dims, *hidden_dims, 1], nonlin=nonlin)
        self.landing_detector = MLP([state_dims, *hidden_dims, 1], nonlin=nonlin)
        self.frame_rewards = MLP([state_dims, *hidden_dims, 1], nonlin=nonlin)

    def forward(self, state):
        crashed = self.crash_detector(state)
        landed = self.landing_detector(state)
        frame_reward = nn.functional.tanh(self.frame_rewards(state))
        return crashed * -1.0 + landed * 1.0 + (frame_reward / 2)

        # mu =
        # phi_one = torch.narrow(phi, last_dim, 0, 1)
        # phi_two = torch.narrow(phi, last_dim, 1, 1)
        # phi_three = torch.narrow(phi, last_dim, 2, 1)
        # one = self.one(inp)
        # two = self.two(inp)
        # return phi_one * one + phi_two * -1.0 + phi_three * (two + 0.2)


class Policy(nn.Module):
    def __init__(self, layers, min=-1.0, max=1.0):
        super().__init__()
        self.mu = MLP(layers)
        # self.scale = nn.Linear(state_dims, 1, bias=False)
        self.min = min
        self.max = max

    def forward(self, state):
        mu = self.mu(state)
        # scale = torch.sigmoid(self.scale(state))
        return ScaledTanhTransformedGaussian(mu, 0.2, min=self.min, max=self.max)


class TransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, dropout=dropout)
        self.outnet = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, inp, hx=None):
        output, hx = self.lstm(inp, hx)
        output = self.outnet(output)
        return output, hx


def reward_mask_f(state, reward, action):
    r = np.concatenate(reward)
    nonzero = r != 0
    p = np.ones_like(r)
    p = p / (r.shape[0] - nonzero.sum())
    p = p * ~nonzero
    i = np.random.choice(r.shape[0], nonzero.sum(), p=p)
    nonzero[i] = True
    return nonzero[:, np.newaxis]


class LunarLanderViz:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

        self.rew_hist = self.fig.add_subplot(211)
        self.rew_hist.hist(np.zeros((1,)), label='reward')
        self.rew_hist.relim()
        self.rew_hist.autoscale_view()
        self.rew_hist.legend()

        self.prew_hist = self.fig.add_subplot(212)
        self.prew_hist.hist(np.zeros((1,)), label='predicted reward')
        self.prew_hist.relim()
        self.prew_hist.autoscale_view()
        self.rew_hist.legend()

        self.fig.canvas.draw()

    def plot_rewards_histogram(self, b, R):
        with torch.no_grad():
            r = np.concatenate([b.trajectories[t][i].reward for t, i in b.index])
            s = np.stack([b.trajectories[t][i].state for t, i in b.index])

            pr = R(torch.from_numpy(s).to(device=args.device))
            pr = pr.cpu().detach().numpy()

            self.rew_hist.clear()
            self.rew_hist.hist(r, bins=100)
            self.rew_hist.set_yscale('log')
            self.rew_hist.relim()
            self.rew_hist.autoscale_view()

            self.prew_hist.clear()
            self.prew_hist.hist(pr, bins=100)
            self.prew_hist.set_yscale('log')
            self.prew_hist.relim()
            self.prew_hist.autoscale_view()

            self.fig.canvas.draw()


def gather_experience(buff, episode, env, policy, eps=0.0, eps_policy=None, render=True, seed=None):
    with torch.no_grad():
        # gather new experience
        episode_reward = 0.0
        if seed is not None:
            env.seed(seed)
        state, reward, done = env.reset(), 0.0, False
        if random() >= eps:
            action = policy(env.connector.policy_prepro(state, args.device).unsqueeze(0)).rsample()
        else:
            action = eps_policy(env.connector.policy_prepro(state, args.device).unsqueeze(0)).rsample()
        action = env.connector.action_prepro(action)
        buff.append(episode,
                    env.connector.buffer_prepro(state),
                    action,
                    env.connector.reward_prepro(reward),
                    done,
                    None)
        episode_reward += reward
        if render:
            env.render()
        while not done:
            state, reward, done, info = env.step(action)
            episode_reward += reward
            if random() >= eps:
                action = policy(env.connector.policy_prepro(state, args.device).unsqueeze(0)).rsample()
            else:
                action = eps_policy(env.connector.policy_prepro(state, args.device).unsqueeze(0)).rsample()
            action = env.connector.action_prepro(action)
            buff.append(episode, env.connector.buffer_prepro(state), action, env.connector.reward_prepro(reward), done,
                        None)
            if render:
                env.render()
    return buff, episode_reward


def gather_seed_episodes(env, seed_episodes):
    buff = Buffer()
    for episode in range(seed_episodes):
        gather_experience(buff, episode, env, env.connector.random_policy, render=False)
    return buff


def main(args):
    # curses
    scr = Curses()

    # monitoring
    recent_reward = deque(maxlen=20)
    wandb.gym.monitor()
    imagine_log_cooldown = wm2.utils.Cooldown(secs=30)
    transition_log_cooldown = wm2.utils.Cooldown(secs=30)
    render_cooldown = wm2.utils.Cooldown(secs=30)

    # viz
    viz = LunarLanderViz()

    # environment
    # env = LinEnv()
    # env = gym.make('Pendulum-v0')
    # env = gym.make('LunarLanderContinuous-v2')
    env = LunarLanderContinuous()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    def normalize_reward(reward):
        return reward / 100.0

    env = gym.wrappers.TransformReward(env, normalize_reward)

    env.connector = LunarLanderConnector

    args.state_dims = env.observation_space.shape[0]
    args.action_dims = env.action_space.shape[0]
    args.action_min = -1.0
    args.action_max = 1.0

    train_buff = gather_seed_episodes(env, args.seed_episodes)
    test_buff = gather_seed_episodes(env, args.seed_episodes)
    train_episode, test_episode = args.seed_episodes, args.seed_episodes

    eps = 0.05

    # policy model
    policy = Policy(layers=[args.state_dims, *args.policy_hidden_dims, args.action_dims], min=args.action_min,
                    max=args.action_max).to(args.device)
    policy_optim = Adam(policy.parameters(), lr=args.lr)

    # value model
    # value = nn.Linear(state_dims, 1)
    value = MLP([args.state_dims, *args.value_hidden_dims, 1], nonlin=args.nonlin).to(args.device)
    value_optim = Adam(value.parameters(), lr=args.lr)

    # transition model
    # T = nn.LSTM(input_size=state_dims + action_dims, hidden_size=state_dims, num_layers=2)
    T = TransitionModel(input_dim=args.state_dims + args.action_dims,
                        hidden_dim=args.transition_hidden_dim, output_dim=args.state_dims,
                        layers=args.transition_layers).to(args.device)
    T_optim = Adam(T.parameters(), lr=args.lr)

    # reward model
    # R = HandcraftedPrior(args.state_dims, args.reward_hidden_dims, nonlin=args.nonlin).to(args.device)
    # R = Mixture(args.state_dims, args.reward_hidden_dims, nonlin=args.nonlin).to(args.device)
    R = MLP([args.state_dims, *args.reward_hidden_dims, 1], nonlin=args.nonlin).to(args.device)
    # R = nn.Linear(state_dims, 1)
    R_optim = Adam(R.parameters(), lr=args.lr)

    # terminal state model
    # D = nn.Linear(state_dims, 1).to(args.device)
    # D_optim = Adam(D.parameters(), lr=args.lr)

    "save and restore helpers"
    R_saver = wm2.utils.SaveLoad('reward')
    policy_saver = wm2.utils.SaveLoad('policy')
    value_saver = wm2.utils.SaveLoad('value')
    T_saver = wm2.utils.SaveLoad('T')

    converged = False

    scr.clear()

    viz.plot_rewards_histogram(train_buff, R)

    while not converged:

        for c in range(args.collect_interval):

            scr.update_progressbar(c)
            scr.update_slot('wandb', f'{wandb.run.name} {wandb.run.project} {wandb.run.id}')

            # Dynamics learning
            train, test = SARNextSubSequenceDataset(train_buff, mask_f=None), SARNextSubSequenceDataset(test_buff, mask_f=None)
            train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

            # train transition model
            for trajectories in train:
                input = torch.cat((trajectories.state, trajectories.action), dim=2).to(args.device)
                T_optim.zero_grad()
                predicted_state, (h, c) = T(input)
                loss = ((trajectories.next_state.to(args.device) - predicted_state) ** 2) * trajectories.pad.to(
                    args.device)
                loss = loss.mean()
                loss.backward()
                clip_grad_norm_(parameters=T.parameters(), max_norm=100.0)
                T_optim.step()
                scr.update_slot('transition_train', f'Transition training loss {loss.item()}')
                wandb.log({'transition_train': loss.item()})
                T_saver.checkpoint(T, T_optim)

            for trajectories in test:
                input = torch.cat((trajectories.state, trajectories.action), dim=2).to(args.device)
                predicted_state, (h, c) = T(input)
                loss = ((trajectories.next_state.to(args.device) - predicted_state) ** 2) * trajectories.pad.to(
                    args.device)
                loss = loss.mean()
                scr.update_slot('transition_test', f'Transition test loss  {loss.item()}')
                wandb.log({'transition_test': loss.item()})
                T_saver.save_if_best(loss, T)
                if transition_log_cooldown():
                    scr.update_table(trajectories.next_state[10:20, 0, :].detach().cpu().numpy().T, h=10,
                                     title='next_state')
                    scr.update_table(predicted_state[10:20, 0, :].detach().cpu().numpy().T, h=14,
                                     title='predicted')
                    scr.update_table(trajectories.action[10:20, 0, :].detach().cpu().numpy().T, h=16,
                                     title='action')

            train, test = SimpleRewardDataset(train_buff), SimpleRewardDataset(test_buff)

            def weights(b, log=False):
                count = {'0.2': 0, '-1': 0, 'other': 0}
                total = 0

                for traj, t in b.index:
                    step = b.trajectories[traj][t]
                    total += 1
                    if step.reward > 0.20:
                        count['0.2'] += 1
                    elif step.reward <= -1.0:
                        count['-1'] += 1
                    else:
                        count['other'] += 1

                probs = {}
                for k, c in count.items():
                    if log:
                        scr.update_slot(f'{k}', f'{k} : {c}')
                    probs[k] = 1 / (3 * (c + eps))

                wghts = []

                for traj, t in b.index:
                    step = b.trajectories[traj][t]
                    if step.reward > 0.20:
                        wghts.append(probs['0.2'])
                    elif step.reward <= -1.0:
                        wghts.append(probs['-1'])
                    else:
                        wghts.append(probs['other'])

                return wghts

            train_weights, test_weights = weights(train_buff, log=True), weights(test_buff)
            train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
            test_sampler = WeightedRandomSampler(test_weights, len(test_weights), replacement=True)
            train = DataLoader(train, batch_size=256, sampler=train_sampler)
            test = DataLoader(test, batch_size=256, sampler=test_sampler)

            for state, reward in train:
                R_optim.zero_grad()
                # dist = R(state.to(args.device))
                # loss = - dist.log_prob(reward.to(args.device)).mean()
                predicted_reward = R(state.to(args.device))
                loss = ((reward.to(args.device) - predicted_reward) ** 2).mean()
                loss.backward()
                R_optim.step()
                scr.update_slot('reward_train', f'Reward train loss {loss.item()}')
                wandb.log({'reward_train': loss.item()})
                R_saver.checkpoint(R, R_optim)

            for state, reward in test:
                # dist = R(state.to(args.device))
                # loss = - dist.log_prob(reward.to(args.device)).mean()
                predicted_reward = R(state.to(args.device))
                loss = ((reward.to(args.device) - predicted_reward) ** 2).mean()
                scr.update_slot('reward_test', f'Reward test loss  {loss.item()}')
                wandb.log({'reward_test': loss.item()})
                R_saver.save_if_best(loss, R)

            # Reward learning
            # train, test = SARDataset(train_buff, mask_f=env.connector.reward_mask_f), \
            #               SARDataset(test_buff, mask_f=env.connector.reward_mask_f)
            # #train, test = SARDataset(train_buff), SARDataset(test_buff)
            # train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            # test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            # for trajectories in train:
            #     mask, pad = trajectories.mask.to(args.device), trajectories.pad.to(args.device)
            #     R_optim.zero_grad()
            #     predicted_reward = R(trajectories.state.to(args.device))
            #     loss = (((trajectories.reward.to(args.device) - predicted_reward) * mask * pad) ** 2).mean()
            #     loss.backward()
            #     R_optim.step()
            #     scr.update_slot('reward_train', f'Reward train loss {loss.item()}')
            #     wandb.log({'reward_train': loss.item()})
            #
            # for trajectories in test:
            #     predicted_reward = R(trajectories.state.to(args.device))
            #     mask, pad = trajectories.mask.to(args.device), trajectories.pad.to(args.device)
            #     loss = (((trajectories.reward.to(args.device) - predicted_reward) * mask * pad) ** 2).mean()
            #     scr.update_slot('reward_test', f'Reward test loss  {loss.item()}')
            #     wandb.log({'reward_test': loss.item()})

            # Terminal state learning
            # train, test = SDDataset(train_buff), SDDataset(test_buff)
            # train_weights, test_weights = train.weights(), test.weights()
            # train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
            # test_sampler = WeightedRandomSampler(test_weights, len(test_weights))
            # train = DataLoader(train, batch_size=32, sampler=train_sampler, drop_last=False)
            # test = DataLoader(test, batch_size=32, sampler=test_sampler, drop_last=False)

            # while pbar.items_processed < len(train) * 2:
            #
            #     for state, done in train:
            #         D_optim.zero_grad()
            #         predicted_done = D(state)
            #         loss = F.binary_cross_entropy_with_logits(predicted_done, done)
            #         loss.backward()
            #         D_optim.step()
            #
            #     for state, done in test:
            #         predicted_done = D(state)
            #         loss = F.binary_cross_entropy_with_logits(predicted_done, done)

            # Behaviour learning
            train = ConcatDataset([SARDataset(train_buff)])
            train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

            for trajectory in train:
                imagine = [torch.cat((trajectory.state, trajectory.action), dim=2).to(args.device)]
                reward = [R(trajectory.state.to(args.device))]
                # done = [D(trajectory.state.to(args.device))]
                v = [value(trajectory.state.to(args.device))]

                # imagine forward here
                for tau in range(args.horizon):
                    state, (h, c) = T(imagine[tau])
                    action = policy(state).rsample()
                    reward += [R(state)]
                    # done += [D(state)]
                    v += [value(state)]
                    imagine += [torch.cat((state, action), dim=2)]

                # VR = torch.mean(torch.stack(reward), dim=0)
                rstack, vstack = torch.stack(reward), torch.stack(v)
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
                if imagine_log_cooldown():
                    rewards = rstack[-1:, :, 0, 0, 0].detach().cpu().numpy()
                    scr.update_table(rewards, title='imagined rewards and values')
                    v = vstack[:, 0, 0, 0].unsqueeze(0).detach().cpu().numpy()
                    scr.update_table(v, h=2)
                    imagined_trajectory = torch.stack(imagine)[:, 0, 0, :].detach().cpu().numpy().T
                    scr.update_table(imagined_trajectory, h=3, title='imagined trajectory')

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

                "backprop loss"
                policy_optim.zero_grad(), value_optim.zero_grad()
                T_optim.zero_grad(), R_optim.zero_grad()  # , D_optim.zero_grad()
                policy_loss = -VL.mean()
                policy_loss.backward()
                clip_grad_norm_(parameters=policy.parameters(), max_norm=100.0)
                policy_optim.step()

                "housekeeping"
                scr.update_slot('policy_loss', f'Policy loss  {policy_loss.item()}')
                wandb.log({'policy_loss': policy_loss.item()})
                policy_saver.checkpoint(policy, policy_optim)

                " regress value against tau ie: the initial estimated value... "
                policy_optim.zero_grad(), value_optim.zero_grad()
                T_optim.zero_grad(), R_optim.zero_grad()  # , D_optim.zero_grad()
                VN = VL.detach().reshape(L * N, -1)
                values = value(trajectory.state.reshape(L * N, -1).to(args.device))
                value_loss = ((VN - values) ** 2).mean() / 2
                value_loss.backward()
                clip_grad_norm_(parameters=value.parameters(), max_norm=100.0)
                value_optim.step()

                "housekeeping"
                scr.update_slot('value_loss', f'Value loss  {value_loss.item()}')
                wandb.log({'value_loss': value_loss.item()})
                value_saver.checkpoint(value, value_optim)
                value_saver.save_if_best(value_loss, value)

        "run the policy on the environment and collect experience"
        sampled_rewards = []
        for _ in range(10):
            train_buff, reward = gather_experience(train_buff, train_episode, env, policy,
                                                   eps=eps, eps_policy=env.connector.random_policy,
                                                   render=render_cooldown())
            wandb.log({'reward': reward})
            sampled_rewards.append(reward)
            train_episode += 1

            recent_reward.append(reward)
            scr.update_slot('eps', f'EPS: {eps}')
            rr = ''
            for reward in recent_reward:
                rr += f' {reward:.5f},'
            scr.update_slot('recent_rewards', 'Recent rewards: ' + rr)

        policy_saver.save_if_best(mean(sampled_rewards), policy, mode='highest')

        viz.plot_rewards_histogram(train_buff, R)

        if random() < 1.0:
            test_buff, reward = gather_experience(test_buff, test_episode, env, policy,
                                                  eps=eps, eps_policy=env.connector.random_policy,
                                                  render=False)
            test_episode += 1

        # boost or decrease exploration if no reward
        if mean(recent_reward) == 0 and eps < 0.5:
            eps += 0.05
        if mean(recent_reward) > 0.5 and eps > 0.05:
            eps -= 0.05

        converged = False


def demo(args):
    env = gym.make(args.env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    def normalize_reward(reward):
        return reward / 100.0

    env = gym.wrappers.TransformReward(env, normalize_reward)

    env.connector = LunarLanderConnector

    args.state_dims = env.observation_space.shape[0]
    args.action_dims = env.action_space.shape[0]
    args.action_min = -1.0
    args.action_max = 1.0

    dummy_buffer = DummyBuffer()

    # policy model
    policy = Policy(layers=[args.state_dims, *args.policy_hidden_dims, args.action_dims], min=args.action_min,
                    max=args.action_max).to(args.device)
    #wandb_run_dir = 'wandb/run-20200501_043058-nzxvviue'
    #wandb_run_dir = 'wandb/dryrun-20200501_182442-d3ydj1o6'
    #wandb_run_dir = 'wandb/run-20200501_185014-axr8ge2o'
    wandb_run_dir = args.demo

    while True:
        load_dict = wm2.utils.SaveLoad.best(wandb_run_dir, 'policy')
        loss = load_dict['loss']
        policy.load_state_dict(load_dict['model'])
        print(f'best_loss {loss}')
        train_buff, reward = gather_experience(dummy_buffer, 0, env, policy,
                                               eps=0.0, eps_policy=env.connector.random_policy,
                                               render=True)


if __name__ == '__main__':
    args = {'seed_episodes': 5,
            'collect_interval': 1,
            'batch_size': 8,
            'device': 'cuda:1',
            'horizon': 15,
            'discount': 0.99,
            'lam': 0.95,
            'lr': 1e-3,
            'policy_hidden_dims': [300],
            'value_hidden_dims': [300],
            'reward_hidden_dims': [300, 300],
            'nonlin': 'nn.ELU',
            'transition_layers': 2,
            'transition_hidden_dim': 64,
            'env': 'LunarLanderContinuous-v3',
            'demo': 'off'
            }

    parser = argparse.ArgumentParser()
    for argument, value in args.items():
        parser.add_argument('--' + argument, type=type(value), required=False, default=value)
    args = parser.parse_args()

    # args = SimpleNamespace(**args)

    if args.demo == 'off':
        wandb.init(config=args)
        curses.wrapper(main(args))
    else:
        demo(args)
