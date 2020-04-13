from types import SimpleNamespace
from collections import deque
from statistics import mean
from random import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import torch.nn.functional as F
import numpy as np
import wandb

import matplotlib
import matplotlib.pyplot as plt

from distributions import TanhTransformedGaussian
from wm2.data.datasets import Buffer, SARDataset, SARNextDataset, SDDataset
from wm2.utils import Pbar
from data.utils import pad_collate_2


class LinEnv:
    def __init__(self):
        self.pos = np.array([0.0], dtype=np.float32)
        self.step_count = 0

    def reset(self):
        self.pos = np.array([0.0], dtype=np.float32)
        self.step_count = 0
        return self.pos.copy()

    def step(self, action):
        self.step_count += 1
        self.pos += action
        pos = self.pos.copy()

        if pos >= 2.0:
            return pos, 1.0, True
        elif self.step_count > 10:
            return pos, 0.0, True
        elif pos <= -2.0:
            return pos, -1.0, True
        else:
            return pos, 0.0, False


def prepro(state):
    return torch.tensor(state).unsqueeze(0)


def random_policy(state):
    return TanhTransformedGaussian(0.0, 0.5)


def gather_experience(buff, episode, env, policy, eps=0.0, eps_policy=None):
    with torch.no_grad():
        # gather new experience
        episode_reward = 0.0
        state, reward, done = env.reset(), 0.0, False
        if random() >= eps:
            action = policy(prepro(state)).rsample().squeeze()
        else:
            action = eps_policy(prepro(state)).rsample().squeeze()
        buff.append(episode, state, action.cpu().numpy(), reward, done, None)
        episode_reward += reward
        while not done:
            state, reward, done = env.step(action.squeeze(0).cpu().numpy())
            if random() >= eps:
                action = policy(prepro(state)).rsample().squeeze()
            else:
                action = eps_policy(prepro(state)).rsample().squeeze()
            buff.append(episode, state, action.cpu().numpy(), reward, done, None)
    return buff, reward


def gather_seed_episodes(env, seed_episodes):
    buff = Buffer()
    for episode in range(seed_episodes):
        gather_experience(buff, episode, env, random_policy)
    return buff


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(1, 1)
        self.scale = nn.Linear(1, 1, bias=False)

    def forward(self, state):
        mu, scale = self.mu(state), torch.sigmoid(self.scale(state))
        return TanhTransformedGaussian(mu, 0.2)


def reward_mask_f(state, reward, action):
    r = np.concatenate(reward)
    nonzero = r != 0
    p = np.ones_like(r)
    p = p / (r.shape[0] - nonzero.sum())
    p = p * ~nonzero
    i = np.random.choice(r.shape[0], nonzero.sum(), p=p)
    nonzero[i] = True
    return nonzero[:, np.newaxis]


def main(args):
    # monitoring
    recent_reward = deque(maxlen=20)

    # visualization
    plt.ion()
    fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    s = torch.linspace(-2.5, 2.5, 20).view(-1, 1)
    z = torch.zeros(20, 1)
    l_actions, = ax1.plot(s, z, 'b-', label='policy(state)')
    l_rewards, = ax2.plot(s, z, 'b-', label='reward(state)')
    l_next_state_0_2, = ax3.plot(s, z, 'b-', label='T(state,0.2)')
    l_next_state_minus_0_2, = ax3.plot(s, z, 'r-', label='T(state,-0.2)')
    l_value, = ax4.plot(s, z, 'b-', label='value(state)')
    ax1.legend(), ax2.legend(), ax3.legend(), ax4.legend()

    # environment
    env = LinEnv()
    train_buff = gather_seed_episodes(env, args.seed_episodes)
    test_buff = gather_seed_episodes(env, args.seed_episodes)
    train_episode, test_episode = args.seed_episodes, args.seed_episodes

    eps = 0.05

    # policy model
    policy = Policy()
    policy.mu.weight.data[0] = -0.1
    policy.mu.bias.data[0] = -0.5
    policy_optim = Adam(policy.parameters(), lr=args.lr)
    # value model
    value = nn.Linear(1, 1)
    value_optim = Adam(value.parameters(), lr=args.lr)

    # transition model
    T = nn.LSTM(input_size=2, hidden_size=1, num_layers=1)
    T_optim = Adam(T.parameters(), lr=args.lr)

    # reward model
    R = nn.Linear(1, 1)
    R_optim = Adam(R.parameters(), lr=args.lr)

    # terminal state model
    D = nn.Linear(1, 1)
    D_optim = Adam(D.parameters(), lr=args.lr)

    converged = False

    while not converged:
        for c in range(args.collect_interval):

            # Dynamics learning
            train, test = SARNextDataset(train_buff, mask_f=None), SARNextDataset(test_buff, mask_f=None)
            train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            pbar = Pbar(items_to_process=args.trajectories_per_pass, train_len=len(train), batch_size=args.batch_size,
                        label='transition')
            while pbar.items_processed < args.trajectories_per_pass:

                # train transition model
                for trajectory in train:
                    input = torch.cat((trajectory.state, trajectory.action.unsqueeze(2)), dim=2)
                    T_optim.zero_grad()
                    predicted_state, (h, c) = T(input)
                    loss = ((trajectory.next_state - predicted_state) ** 2).mean()
                    loss.backward()
                    T_optim.step()
                    pbar.update_train_loss_and_checkpoint(loss, models={'transition': T}, optimizer=T_optim)

                for trajectory in test:
                    input = torch.cat((trajectory.state, trajectory.action.unsqueeze(2)), dim=2)
                    predicted_state, (h, c) = T(input)
                    loss = ((trajectory.next_state - predicted_state) ** 2).mean()
                    pbar.update_test_loss_and_save_model(loss, models={'transition': T})
            pbar.close()

            # Reward learning
            train, test = SARDataset(train_buff, mask_f=reward_mask_f), SARDataset(test_buff, mask_f=reward_mask_f)
            train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            pbar = Pbar(items_to_process=args.trajectories_per_pass, train_len=len(train),
                        batch_size=args.batch_size, label='reward')
            while pbar.items_processed < args.trajectories_per_pass:

                for trajectory in train:
                    R_optim.zero_grad()
                    predicted_reward = R(trajectory.state)
                    loss = (((trajectory.reward - predicted_reward) * trajectory.mask) ** 2).mean()
                    loss.backward()
                    R_optim.step()
                    pbar.update_train_loss_and_checkpoint(loss)

                for trajectory in test:
                    predicted_reward = R(trajectory.state)
                    loss = (((trajectory.reward - predicted_reward) * trajectory.mask) ** 2).mean()
                    pbar.update_test_loss_and_save_model(loss)
            pbar.close()

            # Terminal state learning
            train, test = SDDataset(train_buff), SDDataset(test_buff)
            train_weights, test_weights = train.weights(), test.weights()
            train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
            test_sampler = WeightedRandomSampler(test_weights, len(test_weights))
            train = DataLoader(train, batch_size=32, sampler=train_sampler, drop_last=False)
            test = DataLoader(test, batch_size=32, sampler=test_sampler, drop_last=False)
            pbar = Pbar(items_to_process=len(train) * 2, train_len=len(train),
                        batch_size=args.batch_size, label='terminal')
            while pbar.items_processed < len(train) * 2:

                for state, done in train:
                    D_optim.zero_grad()
                    predicted_done = D(state)
                    loss = F.binary_cross_entropy_with_logits(predicted_done, done)
                    loss.backward()
                    D_optim.step()
                    pbar.update_train_loss_and_checkpoint(loss)

                for state, done in test:
                    predicted_done = D(state)
                    loss = F.binary_cross_entropy_with_logits(predicted_done, done)
                    pbar.update_test_loss_and_save_model(loss)

            pbar.close()

            # Behaviour learning
            train = ConcatDataset([SARDataset(train_buff), SARDataset(test_buff)])
            train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            pbar = Pbar(items_to_process=args.trajectories_per_pass, train_len=len(train),
                        batch_size=args.batch_size, label='behavior')

            while pbar.items_processed < 10:
                for trajectory in train:
                    imagine = [torch.cat((trajectory.state, trajectory.action.unsqueeze(2)), dim=2)]
                    reward = [R(trajectory.state)]
                    done = [D(trajectory.state)]
                    v = [value(trajectory.state)]
                    for tau in range(args.horizon):
                        state, (h, c) = T(imagine[tau])
                        action = policy(state).rsample()
                        reward += [R(state)]
                        done += [D(state)]
                        v += [value(state)]
                        imagine += [torch.cat((state, action), dim=2)]

                    #VR = torch.mean(torch.stack(reward), dim=0)
                    rstack, vstack = torch.stack(reward), torch.stack(v)
                    n = torch.linspace(0.0, args.horizon, args.horizon + 1)
                    discount = torch.empty_like(n).fill_(args.discount).pow(n).view(-1, 1, 1, 1)
                    k = 10
                    r_mask = torch.cat((torch.ones(k), torch.zeros(args.horizon + 1 - k))).view(-1, 1, 1, 1)
                    v_mask = torch.zeros(args.horizon+1).view(-1, 1, 1, 1)
                    v_mask[k] = 1.0
                    VN = (vstack * v_mask + rstack * r_mask) * discount

                    policy_optim.zero_grad(), value_optim.zero_grad()
                    T_optim.zero_grad(), R_optim.zero_grad(), D_optim.zero_grad()
                    # policy_loss = - VR.mean()
                    policy_loss = -VN.mean()
                    policy_loss.backward(retain_graph=True)
                    policy_optim.step()

                    policy_optim.zero_grad(), value_optim.zero_grad()
                    T_optim.zero_grad(), R_optim.zero_grad(), D_optim.zero_grad()
                    value_loss = ((VN - vstack) ** 2).mean() / 2
                    value_loss.backward()
                    value_optim.step()

                    pbar.update_train_loss_and_checkpoint(policy_loss, models={'policy': policy},
                                                          optimizer=policy_optim)

            pbar.close()

            train_buf, reward = gather_experience(train_buff, train_episode, env, policy,
                                                  eps=eps, eps_policy=random_policy)
            recent_reward.append(reward)
            print('')
            print(f'RECENT REWARD: {mean(recent_reward)} MU:{policy.mu.weight.data} BIAS: {policy.mu.bias.data} EPS: {eps}')

            a = policy.mu(s)
            r = R(s)
            v = value(s)

            s_0_2 = torch.cat((s.view(1, -1, 1), torch.full((1, 20, 1), 0.2)), dim=2)
            s_minus_0_2 = torch.cat((s.view(1, -1, 1), torch.full((1, 20, 1), -0.2)), dim=2)
            next_state_0_2, hidden = T(s_0_2)
            next_state_minus_0_2, hidden = T(s_minus_0_2)

            l_actions.set_ydata(a.detach().cpu().numpy())
            l_rewards.set_ydata(r.detach().cpu().numpy())
            l_next_state_0_2.set_ydata(next_state_0_2.detach().cpu().numpy())
            l_next_state_minus_0_2.set_ydata(next_state_minus_0_2.detach().cpu().numpy())
            l_value.set_ydata(v.detach().cpu().numpy())

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            ax3.relim()
            ax3.autoscale_view()
            ax4.relim()
            ax4.autoscale_view()

            fig.canvas.draw()

            print(recent_reward)
            train_episode += 1
            if random() < 0.1:
                test_buff, reward = gather_experience(test_buff, test_episode, env, policy,
                                                      eps=eps, eps_policy=random_policy)

                test_episode += 1

            # boost or decrease exploration if no reward
            if mean(recent_reward) == 0 and eps < 0.5:
                eps += 0.05
            if mean(recent_reward) > 0.5 and eps > 0.05:
                eps -= 0.05

        converged = False


if __name__ == '__main__':
    args = {'seed_episodes': 20,
            'collect_interval': 30,
            'batch_size': 1,
            'trajectories_per_pass': 40,
            'device': 'cuda:0',
            'horizon': 10,
            'discount': 0.99,
            'lr': 1e-3
            }

    wandb.init(config=args)
    args = SimpleNamespace(**args)

    main(args)
