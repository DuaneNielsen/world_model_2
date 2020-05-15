from collections import deque
import collections
from statistics import mean
import random
import curses
import argparse
import time
import pathlib
import yaml
import re
import types

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.distributions import Normal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.functional as F
import torch.backends.cudnn
import torch.cuda
import numpy as np
import wandb
import gym
import gym.wrappers
import pybulletgym

import matplotlib.pyplot as plt

from wm2.distributions import ScaledTanhTransformedGaussian
from wm2.viz import Curses
from wm2.data.datasets import Buffer, SARDataset, SARNextDataset, SimpleRewardDataset, DummyBuffer, \
    SubsetSequenceBuffer
from wm2.utils import Pbar
from wm2.data.utils import pad_collate_2
import wm2.utils
import wm2.env.wrappers
from wm2.env.LunarLander_v3 import LunarLanderConnector
from wm2.env.pybullet import PyBulletConnector


class MLP(nn.Module):
    def __init__(self, layers, nonlin=None, dropout=0.2):
        super().__init__()
        in_dims = layers[0]
        net = []
        # the use of eval here is a security bug
        nonlin = nn.ELU if nonlin is None else eval(nonlin)
        for hidden in layers[1:-1]:
            net += [nn.Linear(in_dims, hidden)]
            net += [nn.Dropout(dropout)]
            net += [nonlin()]
            in_dims = hidden
        net += [nn.Linear(in_dims, layers[-1], bias=False)]
        net += [nn.Dropout(dropout)]

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


class SoftplusMLP(nn.Module):
    def __init__(self, layers, nonlin=None):
        super().__init__()
        self.mlp = MLP(layers, nonlin)

    def forward(self, input):
        return F.softplus(self.mlp(input))


class Policy(nn.Module):
    def __init__(self, layers, min=-1.0, max=1.0, nonlin=None):
        super().__init__()
        self.mu = MLP(layers, nonlin, dropout=0.0)
        self.scale = nn.Linear(layers[0], 1, bias=False)
        self.min = min
        self.max = max

    def forward(self, state):
        mu = self.mu(state)
        scale = torch.sigmoid(self.scale(state)) + 0.1
        return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)


class TransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, dropout=dropout)
        self.outnet = nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=False),
                                    nn.Dropout(dropout))

    def dropout_off(self):
        self._force_dropout(0)

    def dropout_on(self):
        self._force_dropout(self.dropout)

    def _force_dropout(self, dropout):
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

            elif isinstance(module, nn.LSTM):
                module.dropout = dropout

            elif isinstance(module, nn.GRU):
                module.dropout = dropout

    def forward(self, inp, hx=None):
        output, hx = self.lstm(inp, hx)
        output = self.outnet(output)
        return output, hx


class StochasticTransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim * 2, layers, dropout=dropout)
        self.mu = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout))
        self.sig = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout), nn.Sigmoid())

    def forward(self, inp, hx=None):
        hidden, hx = self.lstm(inp, hx)
        mu, sig = hidden.chunk(2, dim=2)
        mu, sig = self.mu(mu), self.sig(sig) + 0.1
        output_dist = Normal(mu, sig)
        return output_dist, hx

    def dropout_off(self):
        self._force_dropout(0)

    def dropout_on(self):
        self._force_dropout(self.dropout)

    def _force_dropout(self, dropout):
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

            elif isinstance(module, nn.LSTM):
                module.dropout = dropout

            elif isinstance(module, nn.GRU):
                module.dropout = dropout


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Dropout(dropout), nn.ELU())
        self.cell = nn.GRUCell(hidden_dim, hidden_dim * 2)
        self.mu = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout), nn.ELU())
        self.sig = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout), nn.Softplus())

    def forward(self, input, hx=None):
        output = []

        for step in input:
            step = self.enc(step)
            hx = self.cell(step, hx)
            output.append(hx)
        hidden = torch.stack(output)
        mu, sig = hidden.chunk(2, dim=2)
        mu, sig = self.mu(mu), self.sig(sig) + 0.1
        return Normal(mu, sig), hidden


def reward_mask_f(state, reward, action):
    r = np.concatenate(reward)
    nonzero = r != 0
    p = np.ones_like(r)
    p = p / (r.shape[0] - nonzero.sum())
    p = p * ~nonzero
    i = np.random.choice(r.shape[0], nonzero.sum(), p=p)
    nonzero[i] = True
    return nonzero[:, np.newaxis]


class HistogramPanel:
    def __init__(self, fig, panels, fig_index, label):
        self.fig = fig
        self.label = label
        self.hist = fig.add_subplot(*panels, fig_index, )
        self.hist.set_title(self.label)
        self.hist.hist(np.zeros((1,)), label=label)
        self.hist.legend(label)
        self.hist.relim()
        self.hist.autoscale_view()
        self.hist.legend()

    def update(self, data, bins=100, yscale='linear'):
        """
        update the histogram
        :param data: numpy array of dim 1
        :return:
        """
        self.hist.clear()
        self.hist.hist(data, bins=bins)
        self.hist.set_title(self.label)
        self.hist.set_yscale(yscale)
        self.hist.relim()
        self.hist.autoscale_view()
        self.fig.canvas.draw()


class PlotPanel:
    def __init__(self, fig, panels, fig_index, label, length=1000):
        self.fig = fig
        self.label = label
        self.plot = fig.add_subplot(*panels, fig_index, )
        self.plot.set_title(self.label)

    def update(self, sequence, label=''):
        self.plot.set_title(self.label)
        self.plot.plot(sequence, label=label)
        self.plot.relim()
        self.plot.autoscale_view()
        self.fig.canvas.draw()

    def reset(self):
        self.plot.clear()
        self.fig.canvas.draw()


class LiveLine:
    def __init__(self, fig, panels, fig_index, label, length=1000):
        self.fig = fig
        self.label = label
        self.live = fig.add_subplot(*panels, fig_index, )
        self.live.set_title(self.label)
        self.rew_live_length = length
        self.dq = deque(maxlen=self.rew_live_length)

    def update(self, data):
        self.live.clear()
        self.dq.append(data)
        y = np.array(self.dq)
        self.live.set_title(self.label)
        self.live.plot(y)
        self.live.relim()
        self.live.autoscale_view()
        self.fig.canvas.draw()

    def reset(self):
        self.live.clear()
        self.dq = deque(maxlen=self.rew_live_length)
        self.fig.canvas.draw()


class Viz:
    def __init__(self, window_title=None):
        plt.ion()
        self.fig = plt.figure(num=None, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k', )
        if window_title:
            self.fig.canvas.set_window_title(window_title)

        panels = (3, 5)
        self.rew_plot = LiveLine(self.fig, panels, 1, label='episode reward')

        self.dynamics_hist = HistogramPanel(self.fig, panels, 2, label='dynamics')
        self.rew_hist = HistogramPanel(self.fig, panels, 3, label='reward')
        self.prew_hist = HistogramPanel(self.fig, panels, 4, label='predicted_reward')
        self.raw_pcont_hist = HistogramPanel(self.fig, panels, 5, label='measured pcont')
        self.est_pcont_hist = HistogramPanel(self.fig, panels, 6, label='est pcont')
        self.value_hist = HistogramPanel(self.fig, panels, 7, label='value')
        self.sampled_value_hist = HistogramPanel(self.fig, panels, 8, label='sampled value')

        self.policy_grad_norm = LiveLine(self.fig, panels, 9, label='policy gradient')

        self.live_value = PlotPanel(self.fig, panels, fig_index=10, label='episode value')
        self.live_pcont = PlotPanel(self.fig, panels, fig_index=11, label='pcont')
        self.exp_rew_vs_actual = PlotPanel(self.fig, panels, fig_index=12, label='reward: exp vs actual')
        self.live_reward = PlotPanel(self.fig, panels, fig_index=13, label='episode reward')
        self.live_dynamics = PlotPanel(self.fig, panels, fig_index=14, label='episode dynamics')

        self.fig.canvas.draw()
        self.samples_in_histogram = 500

    def plot_rewards_histogram(self, b, R):
        with torch.no_grad():
            if len(b.index) < self.samples_in_histogram:
                index = b.index
            else:
                index = random.sample(b.index,self.samples_in_histogram)
            r = np.concatenate([b.trajectories[t][i].reward for t, i in index])
            s = np.stack([b.trajectories[t][i].state for t, i in index])
            R.eval()
            pr = R(torch.from_numpy(s).to(device=args.device))
            pr = pr.cpu().detach().numpy()
            R.train()

            self.rew_hist.update(r, bins=100, yscale='log')
            self.prew_hist.update(pr, bins=100, yscale='log')
            self.fig.canvas.draw()

    def update_rewards(self, reward):
        self.rew_plot.update(reward)

    def _draw_samples(self, b):
        if len(b.index) < self.samples_in_histogram:
            index = b.index
        else:
            index = random.sample(b.index, self.samples_in_histogram)
        return index

    def update_pcont(self, b, pcont):
        with torch.no_grad():
            index = self._draw_samples(b)
            s = np.stack([b.trajectories[t][i].state for t, i in index])
            pcont_raw = np.stack([b.trajectories[t][i].pcont for t, i in index])
            pred_pcont = pcont(torch.from_numpy(s).to(device=args.device))
            self.raw_pcont_hist.update(pcont_raw, bins=100, yscale='log')
            self.est_pcont_hist.update(pred_pcont.cpu().detach().numpy(), bins=100, yscale='log')

    def update_value(self, b, value):
        with torch.no_grad():
            index = self._draw_samples(b)
            s = np.stack([b.trajectories[t][i].state for t, i in index])
            pred_value = value(torch.from_numpy(s).to(device=args.device))
            self.value_hist.update(pred_value.cpu().detach().numpy(), bins=100, yscale='log')

    def update_trajectory_plots(self, value, R, pcont, T, b, trajectory_id):
        with torch.no_grad():
            self.live_value.reset(), self.live_pcont.reset(), self.exp_rew_vs_actual.reset()
            self.live_reward.reset(), self.live_dynamics.reset()
            s = np.stack(i.state for i in b.trajectories[trajectory_id])
            a = np.stack(i.action for i in b.trajectories[trajectory_id])
            r = np.stack(i.reward for i in b.trajectories[trajectory_id])
            p = np.stack(i.pcont for i in b.trajectories[trajectory_id])
            s = torch.from_numpy(s).to(device=args.device)
            a = torch.from_numpy(a).to(device=args.device)
            self.update_episode_reward(R, s, r)
            self.update_episode_pcont(pcont, s, p)
            self.update_episode_expected_reward(T, R, s, a, r)
            self.update_episode_value(value, s)
            self.update_episode_dynamics(T, s, a)

    def update_episode_reward(self, R, s, r):
        with torch.no_grad():
            pr = R(s).squeeze().cpu().numpy()
            self.live_reward.update(pr, 'predicted')
            self.live_reward.update(r, 'received')

    def update_episode_value(self, value, s):
        with torch.no_grad():
            v = value(s).squeeze().cpu().numpy()
            self.live_value.update(v)

    def update_episode_pcont(self, pcont, s, p):
        with torch.no_grad():
            pc = pcont(s).squeeze().cpu().numpy()
            self.live_pcont.update(pc)
            self.live_pcont.update(p)

    def update_episode_expected_reward(self, T, R, s, a, r):
        with torch.no_grad():
            sa = torch.cat((s, a), dim=1)
            pred_next_dist, hx = T(sa.unsqueeze(1))
            next_state = pred_next_dist.loc.squeeze()
            pr = R(next_state).cpu().numpy()
            self.exp_rew_vs_actual.update(pr)
            self.exp_rew_vs_actual.update(r)

    def update_episode_dynamics(self, T, s, a):
        with torch.no_grad():
            sa = torch.cat((s[:-1], a[:-1]), dim=1)
            dist, hx = T(sa.unsqueeze(1))
            lp = torch.exp(dist.log_prob(s[1:].unsqueeze((1)))).squeeze().mean(1)
            lp = lp.squeeze().cpu().numpy()
            self.live_dynamics.update(lp)

    def sample_grad_norm(self, model, sample=0.01):
        if random.random() < sample:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.policy_grad_norm.update(total_norm)

    def update_dynamics(self, b, T, policy):
        with torch.no_grad():
            if len(b.index) < self.samples_in_histogram:
                index = b.index
            else:
                index = random.sample(b.index, self.samples_in_histogram)
            non_terminals = []
            for t, i in index:
                if not b.trajectories[t][i].done:
                    non_terminals.append((t, i))
            s = np.stack([b.trajectories[t][i].state for t, i in non_terminals])
            a = np.stack([b.trajectories[t][i].action for t, i in non_terminals])
            s = torch.from_numpy(s).to(device=args.device)
            a = torch.from_numpy(a).to(device=args.device)
            sa = torch.cat((s, a), dim=1)
            next_s = np.stack([b.trajectories[t][i+1].state for t, i in non_terminals])
            pred_next_dist, hx = T(sa.unsqueeze(0))
            prob_next = pred_next_dist.log_prob(torch.from_numpy(next_s).to(device=args.device)).exp()
            prob_next = prob_next.cpu().detach().numpy()
            prob_next = prob_next.flatten()
            self.dynamics_hist.update(prob_next, bins=100)

    def update_sampled_values(self, values):
        values = np.concatenate(tuple(values))
        self.sampled_value_hist.update(values, yscale='log')

def gather_experience(buff, episode, env, policy, eps=0.0, eps_policy=None, expln_noise=0.0, render=True, seed=None,
                      delay=0.01):
    with torch.no_grad():
        # gather new experience
        episode_reward = 0.0
        if seed is not None:
            env.seed(seed)
        state, reward, done = env.reset(), 0.0, False
        episode_reward += reward

        def get_action(state, reward, done):
            t_state = env.connector.policy_prepro(state, args.device).unsqueeze(0)
            if random.random() >= eps:
                action = policy(t_state).rsample()
                action = Normal(action, expln_noise).sample()
                action = action.clamp(min=-1.0, max=1.0)
            else:
                action = eps_policy(t_state).rsample()
            action = env.connector.action_prepro(action)
            buff.append(episode,
                        env.connector.buffer_prepro(state),
                        action,
                        env.connector.reward_prepro(reward),
                        done,
                        None)
            if render:
                env.render()
                time.sleep(delay)
                #print(reward, state[-2])
            return action

        action = get_action(state, reward, done)

        while not done:
            state, reward, done, info = env.step(action)
            episode_reward += reward
            action = get_action(state, reward, done)

    return buff, episode_reward


def gather_seed_episodes(env, seed_episodes):
    buff = Buffer()
    for episode in range(seed_episodes):
        gather_experience(buff, episode, env, env.connector.random_policy, render=False)
    return buff


def mse_loss(trajectories, predicted_state):
    loss = ((trajectories.next_state.to(args.device) - predicted_state) ** 2) * trajectories.pad.to(
        args.device)
    return loss.mean()


def log_prob_loss_simple(trajectories, predicted_state):
    return - predicted_state.log_prob(trajectories.next_state.to(args.device)).mean()


def log_prob_loss(trajectories, predicted_state):
    prior = Normal(predicted_state.loc[0:-1], predicted_state.scale[0:-1])
    posterior = Normal(predicted_state.loc[1:], predicted_state.scale[1:])
    div = kl_divergence(prior, posterior).mean()
    log_p = predicted_state.log_prob(trajectories.next_state.to(args.device)).mean()
    return div * 1.0 - log_p


def make_env():
    # environment
    env = gym.make(args.env)
    #env = wm2.env.wrappers.ConcatPrev(env)
    #env = wm2.env.wrappers.AddDoneToState(env)
    #env = wm2.env.wrappers.RewardOneIfNotDone(env)
    env = wm2.env.pybullet.PybulletWalkerWrapper(env)
    env.render()

    # def normalize_reward(reward):
    #     return reward / 100.0

    # def increase_negative(reward):
    #     if reward < -0.5:
    #         return - 4 * (reward - 0.5) ** 2
    #     else:
    #         return reward

    # def boost(reward):
    #     return reward * 1000.0
    #


    #env = gym.wrappers.TransformReward(env, alwaysone)

    connector = eval(args.connector)
    env.connector = connector(env.action_space.shape[0])
    # env.connector = LunarLanderConnector
    return env


def determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # curses
    scr = Curses()

    # monitoring
    recent_reward = deque(maxlen=20)
    wandb.gym.monitor()
    imagine_log_cooldown = wm2.utils.Cooldown(secs=30)
    viz_cooldown = wm2.utils.Cooldown(secs=240)
    render_cooldown = wm2.utils.Cooldown(secs=120)
    episode_refresh = wm2.utils.Cooldown(secs=120)

    # viz
    viz = Viz(window_title=f'{wandb.run.project} {wandb.run.id}')

    env = make_env()

    args.state_dims = env.observation_space.shape[0]
    args.action_dims = env.action_space.shape[0]
    args.action_min = -1.0
    args.action_max = 1.0

    train_buff = gather_seed_episodes(env, args.seed_episodes)
    test_buff = gather_seed_episodes(env, args.seed_episodes)
    train_episode, test_episode = args.seed_episodes, args.seed_episodes
    dummy_buffer = DummyBuffer()

    eps = 0.05

    # policy model
    policy = Policy(layers=[args.state_dims, *args.policy_hidden_dims, args.action_dims], min=args.action_min,
                    max=args.action_max, nonlin=args.policy_nonlin).to(args.device)
    policy_optim = Adam(policy.parameters(), lr=args.policy_lr)

    # value model
    # value = nn.Linear(state_dims, 1)
    value = MLP([args.state_dims, *args.value_hidden_dims, 1], nonlin=args.value_nonlin).to(args.device)
    value_optim = Adam(value.parameters(), lr=args.value_lr)

    # transition model
    # T = nn.LSTM(input_size=args.state_dims + args.action_dims, hidden_size=args.state_dims, num_layers=1).to(args.device)
    # T = TransitionModel(input_dim=args.state_dims + args.action_dims,
    #                     hidden_dim=args.transition_hidden_dim, output_dim=args.state_dims,
    #                     layers=args.transition_layers).to(args.device)
    T = StochasticTransitionModel(input_dim=args.state_dims + args.action_dims,
                                  hidden_dim=args.dynamics_hidden_dim, output_dim=args.state_dims,
                                  layers=args.dynamics_layers).to(args.device)
    # T = GRU(input_dim=args.state_dims + args.action_dims,
    #                     hidden_dim=args.transition_hidden_dim, output_dim=args.state_dims,
    #                     layers=args.transition_layers).to(args.device)

    T_optim = Adam(T.parameters(), lr=args.dynamics_lr)
    t_criterion = log_prob_loss

    # reward model
    # R = HandcraftedPrior(args.state_dims, args.reward_hidden_dims, nonlin=args.nonlin).to(args.device)
    # R = Mixture(args.state_dims, args.reward_hidden_dims, nonlin=args.nonlin).to(args.device)
    # R = MLP([args.state_dims, *args.reward_hidden_dims, 1], nonlin=args.reward_nonlin).to(args.device)
    # R = nn.Linear(state_dims, 1)
    # R_optim = Adam(R.parameters(), lr=args.reward_lr)
    class DiffReward(nn.Module):
        def __init__(self):
            super().__init__()
            pass

        def forward(self, state):
            state = torch.transpose(state, 0, -1)
            done_flag = state[-1]
            target_dist = state[-2]
            #target_dist = target_dist.clamp(max=0.999, min=0.0)
            #reward = (1.0 - done_flag + (0.95 ** (target_dist * 1000.0 / 20.0)) * (1.0 - done_flag)).unsqueeze(0)
            # reward = (1.0 - state[-1]) / ((1.0 - state[-2]).sqrt() + torch.finfo(state).eps)
            eps = torch.finfo(target_dist.dtype).eps
            #reward = ((1.5 - torch.log(1.5 - target_dist)) * (1.0 - done_flag)).unsqueeze(0)
            # reward = (1.0 - done_flag)
            #reward = 20 / (1 + torch.exp((target_dist - 0.7) * 10)) * (1.0-done_flag)
            reward = 10 * (1.0 - target_dist) + 0.5
            reward = reward * (1.0 - done_flag)
            reward = reward.unsqueeze(0)
            reward = torch.transpose(reward, 0, -1)
            return reward

    R = DiffReward()

    """ probability of continuing """
    pcont = SoftplusMLP([args.state_dims, *args.pcont_hidden_dims, 1]).to(args.device)
    pcont_optim = Adam(pcont.parameters(), lr=args.pcont_lr)

    "save and restore helpers"
    R_saver = wm2.utils.SaveLoad('reward')
    policy_saver = wm2.utils.SaveLoad('policy')
    value_saver = wm2.utils.SaveLoad('value')
    T_saver = wm2.utils.SaveLoad('T')
    pcont_saver = wm2.utils.SaveLoad('pcont')

    converged = False
    scr.clear()
    best_ave_reward = 0

    while not converged:

        for c in range(args.collect_interval):

            sample_train_buff = SubsetSequenceBuffer(train_buff, args.batch_size, args.horizon + 1)
            sample_test_buff = SubsetSequenceBuffer(test_buff, args.batch_size, args.horizon + 1)

            scr.update_progressbar(c)
            scr.update_slot('wandb', f'{wandb.run.name} {wandb.run.project} {wandb.run.id}')
            scr.update_slot('buffer_stats',
                            f'train_buff size {len(sample_train_buff)} rejects {sample_train_buff.rejects}')

            def train_dynamics():
                # Dynamics learning
                train, test = SARNextDataset(sample_train_buff, mask_f=None), SARNextDataset(sample_test_buff,
                                                                                             mask_f=None)
                train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
                test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

                T.dropout_on()

                # train transition model
                for trajectories in train:
                    input = torch.cat((trajectories.state, trajectories.action), dim=2).to(args.device)
                    T_optim.zero_grad()
                    predicted_state, h = T(input)
                    loss = t_criterion(trajectories, predicted_state)
                    loss.backward()
                    clip_grad_norm_(parameters=T.parameters(), max_norm=100.0)
                    T_optim.step()
                    scr.update_slot('transition_train', f'Transition training loss {loss.item()}')
                    wandb.log({'transition_train': loss.item()})
                    T_saver.checkpoint(T, T_optim)

                T.dropout_off()

                # for trajectories in test:
                #     input = torch.cat((trajectories.state, trajectories.action), dim=2).to(args.device)
                #     predicted_state, h = T(input)
                #     loss = t_criterion(trajectories, predicted_state)
                #     scr.update_slot('transition_test', f'Transition test loss  {loss.item()}')
                #     wandb.log({'transition_test': loss.item()})

            def train_reward():

                # train_weights, test_weights = weights(sample_train_buff, log=True), weights(sample_test_buff)
                # train_sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
                # test_sampler = WeightedRandomSampler(test_weights, len(test_weights), replacement=True)
                # train = DataLoader(train, batch_size=40*50, sampler=train_sampler)
                # test = DataLoader(test, batch_size=40*50, sampler=test_sampler)

                train, test = SimpleRewardDataset(sample_train_buff), SimpleRewardDataset(sample_test_buff)
                train = DataLoader(train, batch_size=args.batch_size * 50, shuffle=True)
                test = DataLoader(test, batch_size=args.batch_size * 50, shuffle=True)

                # R.train()
                #
                # for state, reward in train:
                #     R_optim.zero_grad()
                #     predicted_reward = R(state.to(args.device))
                #     loss = ((reward.to(args.device) - predicted_reward) ** 2).mean()
                #     loss.backward()
                #     R_optim.step()
                #     scr.update_slot('reward_train', f'Reward train loss {loss.item()}')
                #     wandb.log({'reward_train': loss.item()})
                #     R_saver.checkpoint(R, R_optim)
                #
                # R.eval()

                # for state, reward in test:
                #     predicted_reward = R(state.to(args.device))
                #     loss = ((reward.to(args.device) - predicted_reward) ** 2).mean()
                #     scr.update_slot('reward_test', f'Reward test loss  {loss.item()}')
                #     wandb.log({'reward_test': loss.item()})

            def train_pcont():
                """ probability of continuing """
                train, test = SARDataset(sample_train_buff, mask_f=None), SARDataset(sample_test_buff, mask_f=None)
                train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
                test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

                pcont.train()
                for trajectories in train:
                    pcont_optim.zero_grad()
                    predicted_pcont = pcont(trajectories.state.to(args.device))
                    loss = ((predicted_pcont - trajectories.pcont.to(args.device)) ** 2).mean()
                    loss.backward()
                    pcont_optim.step()
                    scr.update_slot('pcont_train', f'pcont train loss  {loss.item()}')
                pcont.eval()

                # for trajectories in test:
                #     predicted_pcont = pcont(trajectories.state.to(args.device))
                #     loss = ((predicted_pcont - trajectories.pcont.to(args.device)) ** 2).mean()
                #     scr.update_slot('pcont_test', f'pcont test loss  {loss.item()}')

            def train_behavior():

                #logging
                update_text = imagine_log_cooldown()
                value_sample = deque(maxlen=100)

                # Behaviour learning
                train = ConcatDataset([SARDataset(sample_train_buff)])
                train = DataLoader(train, batch_size=args.batch_size * 40, collate_fn=pad_collate_2, shuffle=True)

                for trajectory in train:

                    trajectory.state = trajectory.state.to(args.device)
                    trajectory.action = trajectory.action.to(args.device)

                    # anchor on the sampled trajectory
                    imagine = [torch.cat((trajectory.state, trajectory.action), dim=2)]
                    reward = [R(trajectory.state)]
                    p_of_continuing = [pcont(trajectory.state)]
                    v = [value(trajectory.state)]

                    # imagine forward here
                    for tau in range(args.horizon):
                        state, h = T(imagine[tau])
                        if isinstance(state, torch.distributions.Distribution):
                            state = state.rsample()
                        action = policy(state).rsample()
                        reward += [R(state)]
                        p_of_continuing += [pcont(state)]
                        v += [value(state)]
                        imagine += [torch.cat((state, action), dim=2)]

                    # VR = torch.mean(torch.stack(reward), dim=0)
                    rstack, vstack, pcontstack = torch.stack(reward), torch.stack(v), torch.stack(p_of_continuing)
                    vstack = vstack * pcontstack
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
                        rewards = rstack[-1:, :, 0, 0, 0].detach().cpu().numpy()
                        scr.update_table('rewards', rewards)
                        v = vstack[:, 0, 0, 0].unsqueeze(0).detach().cpu().numpy()
                        scr.update_table('values', v)
                        #imagined_trajectory = torch.stack(imagine)[:, 0, 0, :].detach().cpu().numpy().T
                        #scr.update_table('imagined trajectory', imagined_trajectory)

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


                    "backprop loss"
                    policy_optim.zero_grad()#, value_optim.zero_grad()
                    #T_optim.zero_grad(),  pcont_optim.zero_grad(), #R_optim.zero_grad(),
                    policy_loss = -VL.mean()
                    policy_loss.backward()
                    clip_grad_value_(policy.parameters(), 0.001)
                    policy_optim.step()

                    "housekeeping"
                    viz.sample_grad_norm(policy, sample=0.05)
                    scr.update_slot('policy_loss', f'Policy loss  {policy_loss.item()}')
                    wandb.log({'policy_loss': policy_loss.item()})
                    policy_saver.checkpoint(policy, policy_optim)

                    " regress value against tau ie: the initial estimated value... "
                    value.train()
                    value_optim.zero_grad() # policy_optim.zero_grad(),
                    #T_optim.zero_grad(),  #pcont_optim.zero_grad(), #R_optim.zero_grad(),
                    VN = VL.detach()
                    value_sample.append(VN.clone().detach().cpu().flatten().numpy())
                    values = value(trajectory.state)
                    value_loss = ((VN - values) ** 2).mean() / 2
                    value_loss.backward()
                    #clip_grad_norm_(parameters=value.parameters(), max_norm=100.0)
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

            train_dynamics()
            train_reward()
            train_pcont()
            train_behavior()

        "run the policy on the environment and collect experience"
        train_buff, reward = gather_experience(train_buff, train_episode, env, policy,
                                               eps=0.0, eps_policy=env.connector.random_policy, seed=args.seed,
                                               render=render_cooldown(), expln_noise=args.exploration_noise)
        if episode_refresh():
            viz.update_trajectory_plots(value, R, pcont, T, train_buff, train_episode)

        train_episode += 1

        wandb.log({'reward': reward})
        viz.update_rewards(reward)
        recent_reward.append(reward)
        scr.update_slot('eps', f'exploration_noise: {args.exploration_noise}')
        rr = ''
        for reward in recent_reward:
            rr += f' {reward:.5f},'
        scr.update_slot('recent_rewards', 'Recent rewards: ' + rr)
        scr.update_slot('beat_ave_reward', f'Best ave reward: {best_ave_reward}')

        "check if the policy is worth saving"
        if reward > best_ave_reward:
            sampled_rewards = []

            for _ in range(5):
                dummy_buffer, reward = gather_experience(dummy_buffer, train_episode, env, policy,
                                                       eps=0.0, eps_policy=env.connector.random_policy,
                                                       expln_noise=0.0, seed=args.seed)
                sampled_rewards.append(reward)
            if mean(sampled_rewards) > best_ave_reward:
                best_ave_reward = mean(sampled_rewards)
                policy_saver.save(policy, 'best', ave=mean(sampled_rewards), max=max(sampled_rewards),
                                  explr_noise=args.exploration_noise)

        test_buff, reward = gather_experience(test_buff, test_episode, env, policy,
                                              eps=0.0, eps_policy=env.connector.random_policy,
                                              render=False, expln_noise=args.exploration_noise, seed=args.seed)
        test_episode += 1

        if viz_cooldown():
            viz.plot_rewards_histogram(train_buff, R)
            viz.update_dynamics(test_buff, T, env.connector.uniform_random_policy)
            viz.update_pcont(test_buff, pcont)
            viz.update_value(test_buff, value)

        converged = False


def demo(args):
    env = make_env()
    env.render()

    args.state_dims = env.observation_space.shape[0]
    args.action_dims = env.action_space.shape[0]
    args.action_min = -1.0
    args.action_max = 1.0

    dummy_buffer = DummyBuffer()

    # policy model
    policy = Policy(layers=[args.state_dims, *args.policy_hidden_dims, args.action_dims], min=args.action_min,
                    max=args.action_max).to(args.device)

    wandb_run_dir = str(next(pathlib.Path().glob(f'wandb/*{args.demo}')))

    while True:
        load_dict = wm2.utils.SaveLoad.load(wandb_run_dir, 'policy', 'best')
        msg = ''
        for arg, value in load_dict.items():
            if arg != 'model':
                msg += f'{arg}: {value} '
        print(msg)
        policy.load_state_dict(load_dict['model'])
        train_buff, reward = gather_experience(dummy_buffer, 0, env, policy,
                                               eps=0.0, eps_policy=env.connector.random_policy,
                                               render=True, delay=1.0 / args.fps)


if __name__ == '__main__':

    defaults = {
        'env': 'HalfCheetahPyBulletEnv-v0',
        'connector': 'wm2.env.pybullet.PyBulletConnector',
        'seed_episodes': 5,
        'collect_interval': 10,
        'batch_size': 40,
        'device': 'cuda:0',
        'horizon': 15,
        'discount': 0.99,
        'lam': 0.95,
        'exploration_noise': 0.3,

        'dynamics_lr': 1e-4,
        'dynamics_layers': 1,
        'dynamics_hidden_dim': 48,

        'pcont_lr': 1e-4,
        'pcont_hidden_dims': [48, 48],
        'pcont_nonlin': 'nn.ELU',

        'value_lr': 2e-5,
        'value_hidden_dims': [300, 300],
        'value_nonlin': 'nn.ELU',

        'policy_lr': 2e-5,
        'policy_hidden_dims': [48, 48],
        'policy_nonlin': 'nn.ELU',

        'reward_lr': 1e-4,
        'reward_hidden_dims': [300, 300],
        'reward_nonlin': 'nn.ELU',

        'demo': 'off',
        'seed': None,
        'fps': 24,
        'config': None
    }

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

    if args.seed is not None:
        determinism(args.seed)

    if args.demo == 'off':
        wandb.init(config=vars(args))
        curses.wrapper(main(args))
    else:
        demo(args)
