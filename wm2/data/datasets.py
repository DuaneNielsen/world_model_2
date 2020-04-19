import random
from collections import deque
from time import sleep

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical
from torch.utils.data import Dataset
from tqdm import tqdm

from data.utils import SARI, one_hot

class Buffer:
    def __init__(self):
        self.trajectories = []
        self.index = []
        self.rewards_count = 0
        self.done_count = 0
        self.steps_count = 0
        self.action_max = 0

    def append(self, traj_id, state, action, reward, done, info):
        """subclass and override this method to get different buffer write behavior"""
        self._append(traj_id, state, action, reward, done, info)

    def _append(self, traj_id, state, action, reward, done, info):
        """ replaces if traj_id already exists in buffer, else appends"""
        if traj_id >= len(self.trajectories):
            self.trajectories.append([])
        t = len(self.trajectories[traj_id])
        self.trajectories[traj_id].append(SARI(state, action, reward, done, info))
        self.index.append((traj_id, t))
        self.steps_count += 1
        if reward != 0:
            self.rewards_count += 1
        if done:
            self.done_count += 1
        if action > self.action_max:
            self.action_max = action

    def get_step(self, item):
        traj_id, step_id = self.index[item]
        return self.trajectories[traj_id][step_id]

    def get_sequence(self, item, len, pad_mode='copy_first'):
        """returns len items, where the item is at the end of the sequence
        ie: [t3, t4, t5, t6] if item is 6 and len is 4
        fills with the first element of the sequence if no element is available
        """
        traj_id, step_id = self.index[item]
        prefix = []
        for i in range(step_id + 1 - len, step_id + 1):
            if pad_mode == 'copy_first':
                i = max(i, 0)
            prefix.append(self.trajectories[traj_id][i])
        return prefix

    def subsequence_index(self, length):
        """
        builds an index so that any element in the index wll always be followed by at least length steps
        this is done by excluding the tail of the sequence from the index
        ie: [t0, t1, t2, t3, t4] with length 3 will build an index of [t0, t1, t2]
        ensures that sampling the index will always return the start point of valid subsequence of at least length
        :param length: the length of subseqences
        :return: the index
        """
        index = []
        for traj_id, traj in enumerate(self.trajectories):
            end = len(traj) - length + 1
            for s in range(0, end):
                index.append((traj_id, s))
        return index

    def __len__(self):
        return len(self.trajectories)


def mask_all(state, reward, action):
    return np.ones((len(state), 1), dtype=bool)


class SARDataset(Dataset):
    def __init__(self, buffer, mask_f=None):
        super().__init__()
        self.b = buffer
        self.mask_f = mask_f if mask_f is not None else mask_all

    def __len__(self):
        return len(self.b.trajectories)

    def __getitem__(self, item):
        trajectory = self.b.trajectories[item]
        state, reward, action, mask = [], [], [], []
        for step in trajectory:
            state += [step.state]
            reward += [step.reward]
            if isinstance(step.action, int):
                action += [one_hot(step.action, self.b.action_max)]
            else:
                action += [step.action]

        return {'state': np.stack(state),
                'action': np.stack(action),
                'reward': np.stack(reward),
                'mask': self.mask_f(state, reward, action)}


class SARNextDataset(Dataset):
    def __init__(self, buffer, mask_f=None):
        super().__init__()
        self.b = buffer
        self.mask_f = mask_f if mask_f is not None else mask_all

    def __len__(self):
        return len(self.b.trajectories)

    def __getitem__(self, item):
        trajectory = self.b.trajectories[item]
        state, reward, action, mask = [], [], [], []
        for step in trajectory:
            state += [step.state]
            reward += [step.reward]
            if isinstance(step.action, int):
                action += [one_hot(step.action, self.b.action_max)]
            else:
                action += [step.action]

        return {'state': np.stack(state[:-1]),
                'action': np.stack(action[:-1]),
                'reward': np.stack(reward[:-1]),
                'next_state': np.stack(state[1:]),
                'mask': self.mask_f(state[:-1], reward[:-1], action[:-1])}


class SARNextDataset(Dataset):
    def __init__(self, buffer, mask_f=None):
        super().__init__()
        self.b = buffer
        self.mask_f = mask_f if mask_f is not None else mask_all

    def __len__(self):
        return len(self.b.trajectories)

    def __getitem__(self, item):
        trajectory = self.b.trajectories[item]
        state, reward, action, mask = [], [], [], []
        for step in trajectory:
            state += [step.state]
            reward += [step.reward]
            if isinstance(step.action, int):
                action += [one_hot(step.action, self.b.action_max)]
            else:
                action += [step.action]

        return {'state': np.stack(state[:-1]),
                'action': np.stack(action[:-1]),
                'reward': np.stack(reward[:-1]),
                'next_state': np.stack(state[1:]),
                'mask': self.mask_f(state[:-1], reward[:-1], action[:-1])}


class RewDataset:
    def __init__(self, buffer, prefix_len, prefix_mode='stack'):
        """ used to train functions that predict when a reward will be received"""
        self.b = buffer
        self.prefix_len = prefix_len
        self.prefix_mode = prefix_mode

    def __len__(self):
        return sum([len(trajectory) for trajectory in self.b.trajectories])

    def _one_hot(self, reward):
        """
        :return: positive reward : 2
        negative reward: 0
        zero reward 1
        """
        if reward < 0:
            return 0
        elif reward > 0:
            return 2
        else:
            return 1

    def __getitem__(self, item):
        step = self.b.get_step(item)
        sequence = self.b.get_sequence(item, self.prefix_len)
        if self.prefix_mode == 'stack':
            states = np.stack([item.state for item in sequence], axis=0)
        else:
            states = np.concatenate([item.state for item in sequence], axis=0)
        return states, step.reward, self._one_hot(step.reward)

    def weights(self):
        """probabilites to rebalance for sparse rewards"""
        w_rew = 1 / self.b.rewards_count * 0.5
        w_no_rew = 1 / (len(self) - self.b.rewards_count) * 0.5

        weights = []
        for t in self.b.trajectories:
            for step in t:
                if step.has_reward:
                    weights.append(w_rew)
                else:
                    weights.append(w_no_rew)
        return weights


class RewardSubsequenceDataset:
    def __init__(self, buffer, prefix_len):
        """
        used to train functions that predict when a reward will be received
        extracts subsequences from the trjactory of len prefix_len
        such that 50% have reward and 50% have no reward
        """
        self.b = buffer
        self.prefix_len = prefix_len
        self.has_reward = []
        self.no_reward = []
        self._find_subsequences_with_reward()

    def _find_subsequences_with_reward(self):
        for i, (trajectory, t) in enumerate(self.b.index):
            step = self.b.trajectories[trajectory][t]
            if step.reward != 0:
                self.has_reward.append(i)
            else:
                self.no_reward.append(i)

    def __len__(self):
        return len(self.has_reward) * 2

    def __getitem__(self, item):
        if random.random() > 0.5:
            index = self.has_reward[random.randint(0, len(self.has_reward) - 1)]
        else:
            index = self.no_reward[random.randint(0, len(self.no_reward) - 1)]
        step = self.b.get_step(index)
        sequence = self.b.get_sequence(index, self.prefix_len)
        states = np.stack([item.state for item in sequence], axis=0)
        return states, step.reward


class SDDataset:
    def __init__(self, buffer):
        """  Returns the done states and a balanced random sample of the not done states"""
        self.b = buffer
        assert self.b.done_count == len(self.b.trajectories)

    def __len__(self):
        return len(self.b.index)

    def __getitem__(self, item):
        step = self.b.get_step(item)
        return step.state, np.array([step.done]).astype(np.float32)

    def weights(self):
        """probabilites to rebalance for sparse done"""
        w_done = 1 / self.b.done_count * 0.5
        w_not_done = 1 / (len(self) - self.b.done_count) * 0.5

        weights = []
        for t in self.b.trajectories:
            for step in t:
                if step.done:
                    weights.append(w_done)
                else:
                    weights.append(w_not_done)
        return weights


class DoneDataset:
    def __init__(self, buffer, prefix_len, prefix_mode='stack'):
        """ used to train functions that predict the terminal state"""
        self.b = buffer
        self.prefix_len = prefix_len
        self.prefix_mode = prefix_mode
        # ensure we didnt make a mistake counting terminal states
        assert self.b.done_count == len(self.b.trajectories)

    def __len__(self):
        return sum([len(trajectory) for trajectory in self.b.trajectories])

    def __getitem__(self, item):
        step = self.b.get_step(item)
        sequence = self.b.get_sequence(item, self.prefix_len)
        if self.prefix_mode == 'cat':
            states = np.concatenate([item.state for item in sequence], axis=0)
        elif self.prefix_mode == 'stack':
            states = np.stack([item.state for item in sequence], axis=0)
        else:
            raise Exception('prefix_mode is cat or stack')
        return states, np.array([step.done], dtype=np.float32)

    def weights(self):
        """probabilites to rebalance for sparse done"""
        w_done = 1 / self.b.done_count * 0.5
        w_not_done = 1 / (len(self) - self.b.done_count) * 0.5

        weights = []
        for t in self.b.trajectories:
            for step in t:
                if step.done:
                    weights.append(w_done)
                else:
                    weights.append(w_not_done)
        return weights


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
        )
        self.model_tails = nn.ModuleList([
            nn.Sequential(nn.Linear(16, 6)),
            nn.Sequential(nn.Linear(16, 1))
        ])

    def forward(self, state):
        features = self.model(state)
        action, value = self.model_tails[0](features), self.model_tails[1](features)
        return action, value


class PrePro():
    def __init__(self):
        self.history = deque([], maxlen=4)

    def __call__(self, state):
        if len(self.history) < 4:
            for i in range(4):
                self.history.append(state)
        else:
            self.history.append(state)

        return torch.from_numpy(np.concatenate(list(self.history)))


def sample_action(state, policy, prepro, env, eps):
    if random.uniform(0.0, 1.0) > eps:
        # s = torch.from_numpy(state.__array__()).unsqueeze(0).float()
        s = prepro(state).unsqueeze(0)
        action_dist, value = policy(s)
        action = Categorical(logits=action_dist).sample().squeeze().numpy()
    else:
        action = env.action_space.sample()
    return action


def gather_data(episodes, env, render=False):
    buff = Buffer()

    policy = Policy()
    state_dict = torch.load(
        '/home/duane/PycharmProjects/SLM-Lab/data/ppo_shared_pong_2020_02_27_230356/model/ppo_shared_pong_t0_s0_net_model.pt')
    policy.load_state_dict(state_dict)

    # env = wrappers.FrameStack(env, 'concat', 4)
    prepro = PrePro()

    for trajectory in tqdm(range(episodes)):
        state, reward, done, info = env.reset(), 0.0, False, {}
        action = sample_action(state, policy, prepro, env, 0.9)
        buff.append(trajectory, state, action, reward, done, info)

        while not done:
            state, reward, done, info = env.step(action)
            action = sample_action(state, policy, prepro, env, 0.09)
            buff.append(trajectory, state, action, reward, done, info)
            if render:
                sleep(0.04)
                env.render()

    return buff