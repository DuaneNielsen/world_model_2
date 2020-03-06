import gym
import numpy as np
from atariari.benchmark.wrapper import AtariARIWrapper
import wm2.env.wrappers as wrappers
from keypoints.utils import UniImageViewer
import torch
from wm2.models.causal import TemporalConvNet
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import wandb
from tqdm import tqdm
from collections import deque
from torchvision.transforms.functional import to_tensor
from torch.distributions import Categorical
import random
from statistics import mean

viewer = UniImageViewer()


class Pbar:
    def __init__(self, epochs, train_len, batch_size, label):
        """

        :param epochs: number of epochs to train
        :param train_len: length of the training dataset (number of items)
        :param batch_size: items per batch
        :param label: a label to display on the progress bar
        """
        self.bar = tqdm(total=epochs * train_len)
        self.label = label
        self.loss_move_ave = deque(maxlen=20)
        self.test = []
        self.test_loss = 0.0
        self.train_loss = 0.0
        self.batch_size = batch_size

    def update_test_loss(self, loss):
        self.test.append(loss.item())
        self.test_loss = mean(list(self.test))
        self.bar.set_description(f'{self.label} train_loss: {self.train_loss:.6f} test_loss: {self.test_loss:.6f}')

    def update_train_loss(self, loss):
        self.test = []
        self.loss_move_ave.append(loss.item())
        self.train_loss = mean(list(self.loss_move_ave))
        self.bar.update(self.batch_size)
        self.bar.set_description(f'{self.label} train_loss: {self.train_loss:.6f} test_loss: {self.test_loss:.6f}')

    def close(self):
        self.bar.close()


pbar = None


class SAR:
    def __init__(self, state, action, reward, done):
        self.state = state
        self.action = action
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
        self.action_max = 0

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


def one_hot(a, max, dtype=np.float32):
    hot = np.zeros(max + 1, dtype=dtype)
    hot[a] = 1.0
    return hot


class SARDataset(Dataset):
    def __init__(self, buffer):
        super().__init__()
        self.b = buffer

    def __len__(self):
        return len(self.b.trajectories)

    def __getitem__(self, item):
        trajectory = self.b.trajectories[item]
        sequence = [np.concatenate((step.state, one_hot(step.action, self.b.action_max), step.reward), axis=0) for step
                    in trajectory]
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


def split(data, train_len, test_len):
    total_len = train_len + test_len
    train = Subset(data, range(0, train_len))
    test = Subset(data, range(train_len, total_len))
    return train, test


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


def sample_action(state, policy, prepro, env, eps):
    if random.uniform(0.0, 1.0) > eps:
        # s = torch.from_numpy(state.__array__()).unsqueeze(0).float()
        s = prepro(state).unsqueeze(0)
        action_dist, value = policy(s)
        action = Categorical(logits=action_dist).sample().squeeze().numpy()
    else:
        action = env.action_space.sample()
    return action


def gather_data(episodes):
    buff = Buffer()

    policy = Policy()
    state_dict = torch.load(
        '/home/duane/PycharmProjects/SLM-Lab/data/ppo_shared_pong_2020_02_27_230356/model/ppo_shared_pong_t0_s0_net_model.pt')
    policy.load_state_dict(state_dict)

    env = gym.make('PongNoFrameskip-v4')
    env = wrappers.NoopResetEnv(env, noop_max=30)
    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = AtariARIWrapper(env)
    env = wrappers.AtariAriVector(env)
    env = wrappers.FireResetEnv(env)
    # env = wrappers.FrameStack(env, 'concat', 4)
    prepro = PrePro()

    from time import sleep

    for trajectory in tqdm(range(episodes)):
        state, reward, done = env.reset(), 0.0, False
        action = sample_action(state, policy, prepro, env, 0.9)
        buff.append(trajectory, state, action, reward, done)

        while not done:
            state, reward, done, info = env.step(action)
            action = sample_action(state, policy, prepro, env, 0.09)
            buff.append(trajectory, state, action, reward, done)
            if wandb.config.render:
                sleep(0.04)
                env.render()

    return buff


def train_reward(buff, test_buff, epochs, test_freq=2):
    train = RewDataset(buff, prefix_len=wandb.config.rew_prefix_length)
    pbar = Pbar(epochs, len(train), wandb.config.rew_batchsize, "reward")
    weights = train.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    train = DataLoader(train, batch_size=wandb.config.rew_batchsize, sampler=sampler, drop_last=True)

    test = RewDataset(test_buff, prefix_len=wandb.config.rew_prefix_length)
    weights = test.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    test = DataLoader(test, batch_size=wandb.config.rew_batchsize, sampler=sampler, drop_last=True)

    model = nn.Sequential(nn.Linear(wandb.config.rew_prefix_length * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
                          nn.Linear(512, 1, bias=False))
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for source, target in train:
            optim.zero_grad()
            estimate = model(source)
            loss = ((target - estimate) ** 2).mean()
            loss.backward()
            optim.step()
            wandb.log({'reward_loss': loss.item()})
            pbar.update_train_loss(loss)

        if epoch % test_freq == 0:
            with torch.no_grad():
                for source, target in test:
                    estimate = model(source)
                    loss = ((target - estimate) ** 2).mean()
                    wandb.log({'reward_loss': loss.item()})
                    pbar.update_test_loss(loss)
    pbar.close()


def train_done(buff, test_buff, epochs, test_freq):
    train = DoneDataset(buff, prefix_len=wandb.config.done_prefix_len)
    pbar = Pbar(epochs, len(train), wandb.config.rew_batchsize, "done")
    weights = train.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    train = DataLoader(train, batch_size=wandb.config.done_batchsize, sampler=sampler, drop_last=True)

    test = DoneDataset(test_buff, prefix_len=wandb.config.done_prefix_len)
    weights = test.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    test = DataLoader(test, batch_size=wandb.config.done_batchsize, sampler=sampler, drop_last=True)

    model = nn.Sequential(nn.Linear(wandb.config.done_prefix_len * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
                          nn.Linear(512, 1, bias=False))
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for source, target in train:
            optim.zero_grad()
            estimate = model(source)
            loss = criterion(target, estimate)
            loss.backward()
            optim.step()
            wandb.log({'done_loss': loss.item()})
            pbar.update_train_loss(loss)

        if epoch % test_freq == 0:
            with torch.no_grad():
                for source, target in test:
                    estimate = model(source)
                    loss = criterion(target, estimate)
                    wandb.log({'done_loss': loss.item()})
                    pbar.update_test_loss(loss)
    pbar.close()


# def train_done(buff):
#     data = DoneDataset(buff, prefix_len=wandb.config.done_prefix_len)
#     weights = data.weights()
#     sampler = WeightedRandomSampler(weights, len(weights))
#     data = DataLoader(data, batch_size=wandb.config.done_batchsize, sampler=sampler, drop_last=True)
#     criterion = nn.BCEWithLogitsLoss()
#
#     model = nn.Sequential(nn.Linear(wandb.config.done_prefix_len * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
#                           nn.Linear(512, 1, bias=False))
#     optim = torch.optim.Adam(model.parameters(), lr=1e-4)
#
#     for epoch in range(10000):
#         batch = tqdm(data)
#         for source, target in batch:
#             optim.zero_grad()
#             estimate = model(source)
#             loss = criterion(target, estimate)
#             loss.backward()
#             optim.step()
#             wandb.log({'done_loss': loss.item()})
#             batch.set_description(f'done_loss {loss.item()}')


class PadCollate():
    def __init__(self, target_index=None, offset=1, return_loss_mask=False):
        """
        Returns a source signal and copy of the source signal advanced by 1 or more steos
        Used to learn a predictive sequence
        Dimensions (N, C, L) where N = batch size, C is number of input channels, L is the sequence length
        Sequences are padded with zeros so they are all the same length as the longest sequence
        :param index_tensor: if you want to return just a subset of the the input signals as a target,
        specify the indices in a list eg: [0, 1, 3, 4]
        :param offset: the amount to advance the target signal, default 1 timestep
        :param return_loss_mask: if true, returns a lost mask that can be used to ignore padding
        """
        self.index_tensor = torch.LongTensor(target_index) if target_index is not None else None
        self.offset = offset
        self.return_loss_mask = return_loss_mask

    def __call__(self, batch):
        longest = max([traj.shape[1] for traj in batch])
        mask = [np.ones((1, t.shape[1])) for t in batch]
        batch = [np.pad(t, ((0, 0), (0, longest - t.shape[1]))) for t in batch]
        mask = [np.pad(t, ((0, 0), (0, longest - t.shape[1]))) for t in mask]
        batch = np.stack(batch, axis=0)

        source = np.pad(batch.copy(), ((0, 0), (0, 0), (self.offset, 0)))
        target = np.pad(batch, ((0, 0), (0, 0), (0, self.offset)))
        source = torch.from_numpy(source)
        if self.index_tensor is not None:
            target = torch.from_numpy(target)[:, self.index_tensor, :]
        else:
            target = torch.from_numpy(target)
        if self.return_loss_mask:
            mask = np.stack(mask, axis=0)
            # right padding is not useful
            mask = np.pad(mask, ((0, 0), (0, 0), (0, self.offset)))
            # discard left padding also
            mask[:, :, 0] = 0
            mask = torch.from_numpy(mask)
            return source, target, mask
        return source, target


def train_predictor(predictor, train_buff, test_buff, indices, epochs, label, batch_size, test_freq=2):

    train, test = SARDataset(train_buff), SARDataset(test_buff)
    pbar = Pbar(epochs=epochs, train_len=len(train), batch_size=batch_size, label=label)
    pad_collate = PadCollate(indices, return_loss_mask=True)
    train = DataLoader(train, batch_size=batch_size, collate_fn=pad_collate)
    test = DataLoader(test, batch_size=wandb.config.test_len, collate_fn=pad_collate)

    optim = Adam(predictor.parameters(), lr=1e-4)

    for epoch in range(epochs):

        for source, target, mask in train:
            source, target, mask = source.to(device), target.to(device), mask.to(device)
            optim.zero_grad()
            estimate = predictor(source)
            loss = (((target - estimate) * mask) ** 2).mean()
            loss.backward()
            optim.step()
            wandb.log({f'{label}_pred_loss': loss.item()})
            pbar.update_train_loss(loss)

        if epoch % test_freq == 0:
            with torch.no_grad():
                for source, target, mask in test:
                    source, target, mask = source.to(device), target.to(device), mask.to(device)
                    estimate = predictor(source)
                    loss = (((target - estimate) * mask) ** 2).mean()
                    wandb.log({f'{label}_pred_test_loss': loss.item()})
                    pbar.update_test_loss(loss)

    pbar.close()


def main():
    player = [0]
    enemy = [1]
    ball = [2, 3]
    all = [0, 1, 2, 3]

    buff = gather_data(wandb.config.train_len)
    test_buff = gather_data(wandb.config.test_len)

    train_reward(buff, test_buff, epochs=10, test_freq=3)
    train_done(buff, test_buff, epochs=10, test_freq=3)

    player_predictor = TemporalConvNet(11, [512, 512, 512, 512, len(player)]).to(device)
    train_predictor(predictor=player_predictor, test_buff=buff, train_buff=test_buff, indices=player, epochs=10,
                    label='player', batch_size=wandb.config.predictor_batchsize)

    ball_predictor = TemporalConvNet(11, [512, 512, 512, 512, len(ball)]).to(device)
    train_predictor(predictor=ball_predictor, test_buff=buff, train_buff=test_buff, indices=ball, epochs=20,
                    label='ball', batch_size=wandb.config.predictor_batchsize)

    enemy_predictor = TemporalConvNet(11, [512, 512, 512, 512, len(enemy)]).to(device)
    train_predictor(predictor=enemy_predictor, test_buff=buff, train_buff=test_buff, indices=enemy, epochs=10,
                    label='enemy', batch_size=wandb.config.predictor_batchsize)

    all_predictor = TemporalConvNet(11, [512, 512, 512, 512, len(all)]).to(device)
    train_predictor(predictor=all_predictor, test_buff=buff, train_buff=test_buff, indices=all, epochs=20,
                    label='all', batch_size=wandb.config.predictor_batchsize)


if __name__ == '__main__':
    defaults = dict(frame_op_len=8,
                    train_len=32,
                    test_len=16,
                    predictor_batchsize=32,
                    render=False,
                    rew_prefix_length=4,
                    rew_batchsize=32,
                    done_prefix_len=4,
                    done_batchsize=32,
                    test_freq=10)

    dev = dict(frame_op_len=8,
               train_len=2,
               test_len=2,
               predictor_batchsize=2,
               render=False,
               rew_prefix_length=4,
               rew_batchsize=256,
               done_prefix_len=4,
               done_batchsize=256,
               test_freq=10)

    wandb.init(config=dev)

    # config validations
    config = wandb.config
    assert config.predictor_batchsize <= config.train_len
    assert config.test_len >= 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()
