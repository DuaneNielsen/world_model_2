import gym
import numpy as np
from atariari.benchmark.wrapper import AtariARIWrapper
import wm2.env.wrappers as wrappers
from keypoints.utils import UniImageViewer
import torch

from utils import TensorNamespace, Pbar, SARI, one_hot, gaussian_like_function, debug_image
from wm2.models.causal import Model
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import wandb
from tqdm import tqdm
from collections import deque
from torch.distributions import Categorical
import random
from time import sleep
from types import SimpleNamespace
import pickle
from pathlib import Path

viewer = UniImageViewer()


class Buffer:
    def __init__(self):
        self.trajectories = []
        self.index = []
        self.rewards_count = 0
        self.done_count = 0
        self.steps_count = 0
        self.action_max = 0

    def append(self, traj_id, state, action, reward, done, info):
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

    def __len__(self):
        return len(self.trajectories)


class SARDataset(Dataset):
    def __init__(self, buffer):
        super().__init__()
        self.b = buffer

    def __len__(self):
        return len(self.b.trajectories)

    def __getitem__(self, item):
        trajectory = self.b.trajectories[item]
        state, reward, action = [], [], []
        for step in trajectory:
            state += [step.state]
            reward += [step.reward]
            action += [one_hot(step.action, self.b.action_max)]

        return {'state': np.stack(state), 'action': np.stack(action), 'reward': np.stack(reward)}


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


def chomp(seq, end, dim, bite_size):
    chomped_len = seq.size(dim) - bite_size
    if end == 'left':
        return torch.narrow(seq, dim, bite_size, chomped_len)
    if end == 'right':
        return torch.narrow(seq, dim, 0, chomped_len)
    else:
        Exception('end parameter must be left or right')


def pad(batch, longest):
    params = {}
    for trajectory in batch:
        pad_len = longest - trajectory['state'].shape[0]
        padding = [(0, pad_len)]  # right pad only the first dim
        padding += [(0, 0) for _ in range(len(trajectory['state'].shape)-1)]

        for key in trajectory:
            if key not in params:
                params[key] = []
            params[key] += [np.pad(trajectory[key], padding)]

    for key in params:
        params[key] = np.stack(params[key])
    return params


def make_mask(batch, longest, dtype=np.float):
    mask = []
    for trajectory in batch:
        l = trajectory['state'].shape[0]
        mask += [np.concatenate((np.ones(l, dtype=dtype), np.zeros(longest - l, dtype=dtype)))]
    mask = np.stack(mask)
    mask = np.expand_dims(mask, axis=2)
    return mask


def pad_collate(batch):
    longest = max([trajectory['state'].shape[0] for trajectory in batch])
    data = pad(batch, longest)
    dtype = data[next(iter(data))].dtype
    mask = make_mask(batch, longest, dtype=dtype)
    data['mask'] = mask
    for key in data:
        data[key] = torch.from_numpy(data[key])
    return TensorNamespace(**data)


def autoregress(state, action, reward, mask, target_start=0, target_length=None, target_reward=False, advance=1):
    """

    :param state: (N, T, S)
    :param action: (N, T, A)
    :param reward: (N, T, 1)
    :param mask: (N, T, 1)
    :param target_start: start index of a slice across the state dimension to output as target
    :param target_length: length of slice across the state dimension to output as target
    :param target_reward: outputs reward as the target
    :param advance: the amount of timesteps to advance the target, default 1
    :return:
    source: concatenated (state, action, reward),
    target: subset of the source advanced by 1,
    mask: loss_mask that is zero where padding was put, or where it makes no sense to make a prediction
    """
    source = torch.cat((state, reward), dim=2)
    if target_reward:
        target = reward
    else:
        target = state.clone()
    if target_length is not None:
        target = target.narrow(dim=2, start=target_start, length=target_length)

    ret = {}
    ret['source'] = chomp(source, 'right', dim=1, bite_size=advance)
    ret['action'] = chomp(action, 'right', dim=1, bite_size=advance)
    ret['target'] = chomp(target, 'left', dim=1, bite_size=advance)
    ret['mask'] = chomp(mask, 'left', dim=1, bite_size=advance)
    return TensorNamespace(**ret)




def train_predictor(predictor, train_buff, test_buff, epochs, target_start, target_len, label, batch_size, test_freq=50):
    train, test = SARDataset(train_buff), SARDataset(test_buff)
    pbar = Pbar(epochs=epochs, train_len=len(train), batch_size=batch_size, label=label)
    train = DataLoader(train, batch_size=batch_size, collate_fn=pad_collate)
    test = DataLoader(test, batch_size=wandb.config.test_len, collate_fn=pad_collate)

    optim = Adam(predictor.parameters(), lr=1e-4)

    for epoch in range(epochs):

        for mb in train:
            seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, target_start, target_len).to(device)
            optim.zero_grad()
            estimate = predictor(seqs.source, seqs.action)
            loss = (((seqs.target - estimate) * seqs.mask) ** 2).mean()
            loss.backward()
            optim.step()
            pbar.update_train_loss(loss, model=predictor)

        if epoch % test_freq == 0:
            with torch.no_grad():
                for mb in test:
                    seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, target_start, target_len).to(device)
                    estimate = predictor(seqs.source, seqs.action)
                    loss = (((seqs.target - estimate) * seqs.mask) ** 2).mean()
                    pbar.update_test_loss(loss, model=predictor)
                    for step in estimate[0][0:160].cpu().numpy():

                        if label == 'all':
                            pos = np.array([[0.0, 0.2], [0.0, 0.8], [0.0, 0.0]])
                            pos[0, 0] = step[0]
                            pos[1, 0] = step[1]
                            pos[2] = step[2:4]
                        elif label == 'player':
                            pos = np.array([[0.0, 0.8]])
                            pos[0, 0] = step[0]
                        elif label == 'enemy':
                            pos = np.array([[0.0, 0.2]])
                            pos[0, 0] = step[0]
                        elif label == 'ball':
                            pos = np.expand_dims(step, axis=0)

                        probmap = gaussian_like_function(pos, 800, 800, sigma=0.2)
                        image = (probmap * 255).astype(np.uint)
                        debug_image(image, block=False)

    pbar.close()


def load_or_generate(env, n, path=None):
    if path is not None:
        buff_path = Path(path)
        if buff_path.exists():
            with buff_path.open('rb') as f:
                buff = pickle.load(f)
                if len(buff) == n:
                    return buff

    buff = gather_data(n, env, wandb.config.render)
    if path is not None:
        with Path(path).open('wb') as f:
            pickle.dump(buff, f)
    return buff

def main():

    env = gym.make('PongNoFrameskip-v4')
    env = wrappers.NoopResetEnv(env, noop_max=30)
    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = AtariARIWrapper(env)
    env = wrappers.AtariAriVector(env)
    env = wrappers.FireResetEnv(env)

    train_buff = load_or_generate(env, wandb.config.train_len, 'data/buff.pkl')
    test_buff = load_or_generate(env, wandb.config.test_len, 'data/test_buff.pkl')


    state_dims = 4
    action_dims = 6
    reward_dims = 1

    ensemble = {}
    ensemble['all'] = SimpleNamespace(hidden=[512, 512, 512, 512], epochs=1000, target_start=0, target_len=4)
    ensemble['player'] = SimpleNamespace(hidden=[512, 512, 512, 512], epochs=500, target_start=0, target_len=1)
    ensemble['enemy'] = SimpleNamespace(hidden=[512, 512, 512, 512], epochs=1000, target_start=1, target_len=1)
    ensemble['ball'] = SimpleNamespace(hidden=[512, 512, 512, 512], epochs=1000, target_start=2, target_len=2)

    # train_reward(buff, test_buff, epochs=10, test_freq=3)
    # train_done(buff, test_buff, epochs=10, test_freq=3)

    for label, args in ensemble.items():
        predictor = Model(state_dims, action_dims, reward_dims, args.hidden, args.target_len).to(device)
        train_predictor(predictor=predictor, train_buff=train_buff, test_buff=test_buff, epochs=args.epochs,
                        target_start=args.target_start,
                        target_len=args.target_len, label=label, batch_size=wandb.config.predictor_batchsize)


if __name__ == '__main__':
    defaults = dict(frame_op_len=8,
                    train_len=64,
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

    wandb.init(config=defaults)

    # config validations
    config = wandb.config
    assert config.predictor_batchsize <= config.train_len
    assert config.test_len >= 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()
