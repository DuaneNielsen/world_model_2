import gym
import numpy as np
from atariari.benchmark.wrapper import AtariARIWrapper
import wm2.env.wrappers as wrappers
from keypoints.utils import UniImageViewer
import torch
from wm2.models.causal import TemporalConvNet
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import wandb
from tqdm import tqdm

viewer = UniImageViewer()


class SAR:
    def __init__(self, state, action, reward, done):
        self.state = state
        self.action = np.array([action], dtype=state.dtype)
        self.reward = np.array([reward], dtype=state.dtype)
        self.has_reward = reward != 0
        self.terminal = done


class Buffer:
    def __init__(self, episodes):
        self.trajectories = [[] for _ in range(episodes)]
        self.index = []
        self.rewards_count = 0
        self.steps_count = 0

    def append(self, traj_id, state, action, reward, done):
        t = len(self.trajectories[traj_id])
        self.trajectories[traj_id].append(SAR(state, action, reward, done))
        self.index.append((traj_id, t))
        self.steps_count += 1
        if reward != 0:
            self.rewards_count += 1

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


class SARDataset:
    def __init__(self, buffer):
        self.b = buffer

    def __len__(self):
        return len(self.b.trajectories)

    def __getitem__(self, item):
        trajectory = self.b.trajectories[item]
        sequence = [np.concatenate((step.state, step.action, step.reward), axis=0) for step in trajectory]
        return np.stack(sequence, axis=1)


class RewDataset:
    def __init__(self, buffer, len):
        """ returns individual steps"""
        self.b = buffer
        self.len = len

    def __len__(self):
        return sum([len(trajectory) for trajectory in self.b.trajectories])

    def __getitem__(self, item):
        step = self.b.get_step(item)
        sequence = self.b.get_sequence(item, self.len)
        states = np.concatenate([item.state for item in sequence], axis=0)
        return states, step.reward

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


def make_env():
    env = gym.make('PongNoFrameskip-v4')
    env = wrappers.NoopResetEnv(env, noop_max=30)
    env = wrappers.MaxAndSkipEnv(env, skip=wandb.config.frame_op_len)
    env = AtariARIWrapper(env)
    env = wrappers.AtariAriVector(env)
    return env


def gather_data(env, episodes):
    buff = Buffer(episodes)

    for trajectory in tqdm(range(episodes)):
        state, reward, done = env.reset(), 0.0, False
        action = env.action_space.sample()
        buff.append(trajectory, state, action, reward, done)

        while not done:
            state, reward, done, info = env.step(action)
            action = env.action_space.sample()
            buff.append(trajectory, state, action, reward, done)
            env.render()

    return buff


def train_reward(buff):
    data = RewDataset(buff, len=wandb.config.rew_length)
    weights = data.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    data = DataLoader(data, batch_size=wandb.config.rew_batchsize, sampler=sampler)

    model = nn.Sequential(nn.Linear(wandb.config.rew_length * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1, bias=False))
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10000):
        batch = tqdm(data)
        for source, target in batch:
            optim.zero_grad()
            estimate = model(source)
            loss = ((target - estimate) ** 2).mean()
            loss.backward()
            optim.step()
            wandb.log({'reward_loss': loss.item()})
            batch.set_description(f'reward_loss {loss.item()}')


def train_state_predictor(buff):
    state_predictor = TemporalConvNet(6, [512, 512, 4])
    state_optim = Adam(state_predictor.parameters(), lr=1e-4)
    dataset = SARDataset(buff)

    for epoch in range(10000):
        for step in range(len(dataset)):
            trajectory = torch.from_numpy(dataset[step])
            # pad by 1 timstep
            pad = torch.zeros(trajectory.size(0), 1)
            source = torch.cat((pad, trajectory), dim=1).unsqueeze(0)
            target = torch.cat((trajectory.clone(), pad), dim=1)
            target_state = target[torch.tensor([0, 1, 2, 3]), :].unsqueeze(0)

            state_optim.zero_grad()
            estimate = state_predictor(source)
            loss = ((target_state - estimate) ** 2).mean()
            loss.backward()
            state_optim.step()
            state_loss = loss.item()

            print(f'state_loss {state_loss} ')


def main():
    env = make_env()
    buff = gather_data(env, wandb.config.episodes)
    train_reward(buff)
    # train_state_predictor(buff)


if __name__ == '__main__':
    defaults = dict(frame_op_len=8,
                    episodes=3,
                    rew_length=4,
                    rew_batchsize=32)

    wandb.init(config=defaults)

    main()
