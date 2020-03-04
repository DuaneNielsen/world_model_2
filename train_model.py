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
from collections import deque
from torchvision.transforms.functional import to_tensor
from torch.distributions import Categorical

viewer = UniImageViewer()

class SAR:
    def __init__(self, state, action, reward, done):
        self.state = state
        self.action = np.array([action], dtype=state.dtype)
        self.reward = np.array([reward], dtype=state.dtype)
        self.has_reward = reward != 0
        self.done = done


class Buffer:
    def __init__(self):
        self.trajectories = []
        self.index = []
        self.rewards_count = 0
        self.done_count = 0
        self.steps_count = 0

    def append(self, traj_id, state, action, reward, done):
        if traj_id >= len(self.trajectories):
            self.trajectories.append([])
        t = len(self.trajectories[traj_id])
        self.trajectories[traj_id].append(SAR(state, action, reward, done))
        self.index.append((traj_id, t))
        self.steps_count += 1
        if reward != 0:
            self.rewards_count += 1
        if done:
            self.done_count += 1

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
    def __init__(self, buffer, prefix_len):
        """ used to train functions that predict when a reward will be received"""
        self.b = buffer
        self.prefix_len = prefix_len

    def __len__(self):
        return sum([len(trajectory) for trajectory in self.b.trajectories])

    def __getitem__(self, item):
        step = self.b.get_step(item)
        sequence = self.b.get_sequence(item, self.prefix_len)
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


class DoneDataset:
    def __init__(self, buffer, prefix_len):
        """ used to train functions that predict the terminal state"""
        self.b = buffer
        self.prefix_len = prefix_len
        # ensure we didnt make a mistake counting terminal states
        assert self.b.done_count == len(self.b.trajectories)

    def __len__(self):
        return sum([len(trajectory) for trajectory in self.b.trajectories])

    def __getitem__(self, item):
        step = self.b.get_step(item)
        sequence = self.b.get_sequence(item, self.prefix_len)
        states = np.concatenate([item.state for item in sequence], axis=0)
        return states, np.array([step.done], dtype=np.float32)

    def weights(self):
        """probabilites to rebalance for sparse rewards"""
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



"""
MLPNet(
  (model): Sequential(
    (0): Linear(in_features=16, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=16, bias=True)
    (3): ReLU()
    (4): Linear(in_features=16, out_features=16, bias=True)
    (5): ReLU()
    (6): Linear(in_features=16, out_features=16, bias=True)
    (7): ReLU()
  )
  (model_tails): ModuleList(
    (0): Sequential(
      (0): Linear(in_features=16, out_features=6, bias=True)
    )
    (1): Sequential(
      (0): Linear(in_features=16, out_features=1, bias=True)
    )
  )
  (loss_fn): MSELoss()
)
"""


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

def gather_data(episodes):
    buff = Buffer()

    policy = Policy()
    state_dict = torch.load('/home/duane/PycharmProjects/SLM-Lab/data/ppo_shared_pong_2020_02_27_230356/model/ppo_shared_pong_t0_s0_net_model.pt')
    policy.load_state_dict(state_dict)

    env = gym.make('PongNoFrameskip-v4')
    env = wrappers.NoopResetEnv(env, noop_max=30)
    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = AtariARIWrapper(env)
    env = wrappers.AtariAriVector(env)
    env = wrappers.FireResetEnv(env)
    env = wrappers.FrameStack(env, 'concat', 4)
    #prepro = PrePro()

    for trajectory in tqdm(range(episodes)):
        state, reward, done = env.reset(), 0.0, False
        s = torch.from_numpy(state.__array__()).unsqueeze(0).float()
        #s = prepro(state).unsqueeze(0)
        action_dist, value = policy(s)
        action = Categorical(logits=action_dist).sample().squeeze().numpy()
        #action = env.action_space.sample()
        #action = torch.argmax(action_dist).item()
        #buff.append(trajectory, state, action, reward, done)

        while not done:
            state, reward, done, info = env.step(action)
            s = torch.from_numpy(state.__array__()).unsqueeze(0).float()
            #action = env.action_space.sample()
            #s = prepro(state).unsqueeze(0)
            action_dist, value = policy(s)
            action = Categorical(logits=action_dist).sample().squeeze().numpy()
            # action = env.action_space.sample()
            #action = torch.argmax(action_dist).item()
            #buff.append(trajectory, state, action, reward, done)
            if wandb.config.render:
                env.render()

    return buff


def train_reward(buff):
    data = RewDataset(buff, prefix_len=wandb.config.rew_prefix_length)
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


def train_done(buff):
    data = DoneDataset(buff, prefix_len=wandb.config.done_prefix_len)
    weights = data.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    data = DataLoader(data, batch_size=wandb.config.done_batchsize, sampler=sampler, drop_last=True)
    criterion = nn.BCEWithLogitsLoss()

    model = nn.Sequential(nn.Linear(wandb.config.done_prefix_len * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1, bias=False))
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10000):
        batch = tqdm(data)
        for source, target in batch:
            optim.zero_grad()
            estimate = model(source)
            #loss = ((target - estimate) ** 2).mean()
            loss = criterion(target, estimate)
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
    buff = gather_data(wandb.config.episodes)
    # train_reward(buff)
    train_done(buff)
    # train_state_predictor(buff)


if __name__ == '__main__':
    defaults = dict(frame_op_len=8,
                    episodes=32,
                    render=True,
                    rew_prefix_length=4,
                    rew_batchsize=32,
                    done_prefix_len=4,
                    done_batchsize=32)

    wandb.init(config=defaults)

    main()
