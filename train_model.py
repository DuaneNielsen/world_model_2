from math import pi
import random
from time import sleep
from types import SimpleNamespace
import pickle
from pathlib import Path
from math import floor

import gym
import numpy as np
from atariari.benchmark.wrapper import AtariARIWrapper
import wm2.env.wrappers as wrappers
from keypoints.utils import UniImageViewer
import torch

from utils import TensorNamespace, Pbar, SARI, one_hot, gaussian_like_function, debug_image
from wm2.models.causal import Causal
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import wandb
from tqdm import tqdm
from collections import deque
from torch.distributions import Categorical
from torch.distributions import Normal, LogNormal
from torch.nn.utils import clip_grad_norm_
import utils
import models.causal
import torch.nn.functional as F
from matplotlib import pyplot as plt

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


def multivariate_diag_gaussian(mu, stdev, size):
    """

    :param mu: mean (N, D)
    :param stdev: std-deviation (not variance!) (N, D)
    :param size: tuple of output dimension sizes eg: (h, w)
    :return: a heightmap
    """
    N = mu.size(0)
    D = mu.size(1)
    axes = [torch.linspace(0, 1.0, size[i], dtype=mu.dtype, device=mu.device) for i in range(D)]
    grid = torch.meshgrid(axes)
    grid = torch.stack(grid)
    gridshape = grid.shape[1:]
    grid = grid.flatten(start_dim=1)
    eps = torch.finfo(mu.dtype).eps
    stdev[stdev < eps] = stdev[stdev < eps] + eps

    constant = torch.tensor((2 * pi) ** D).sqrt() * torch.prod(stdev, dim=1)
    numerator = (grid.view(1, D, -1) - mu.view(N, D, 1)) ** 2
    denominator = 2 * stdev ** 2
    exponent = - torch.sum(numerator / denominator.view(N, D, 1), dim=1)
    exponent = torch.exp(exponent)
    height = exponent / constant
    return height.reshape(N, *gridshape)


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
            pbar.update_train_loss_and_checkpoint(loss)

        if epoch % test_freq == 0:
            with torch.no_grad():
                for source, target in test:
                    estimate = model(source)
                    loss = ((target - estimate) ** 2).mean()
                    wandb.log({'reward_loss': loss.item()})
                    pbar.update_test_loss_and_save_model(loss)
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
            pbar.update_train_loss_and_checkpoint(loss)

        if epoch % test_freq == 0:
            with torch.no_grad():
                for source, target in test:
                    estimate = model(source)
                    loss = criterion(target, estimate)
                    wandb.log({'done_loss': loss.item()})
                    pbar.update_test_loss_and_save_model(loss)
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
        padding += [(0, 0) for _ in range(len(trajectory['state'].shape) - 1)]

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

def display_predictions(trajectory_mu, trajectory_stdev, label, max_length=160):
    length = min(trajectory_mu.size(0), max_length)
    for mu, stdev in zip(trajectory_mu[0:length], trajectory_stdev[0:length]):
        image_size = (240 * 4, 160 * 4)

        def strip_index(center, thickness, length):
            center = floor(center * length)
            thickness = floor(thickness * length // 2)
            lower, upper = center - thickness, center + thickness
            return lower, upper

        def put_strip(mu, stdev, center, thickness, image_size, dim):
            image = np.zeros(image_size)
            #stdev = torch.tensor([[0.05]], device=device)
            strip = multivariate_diag_gaussian(mu.view(1, -1), stdev.view(1, -1), image_size)
            strip = strip.cpu().numpy()
            lower, upper = strip_index(center, thickness, image_size[dim])
            if dim == 1:
                image[:, lower:upper] = strip.T
            elif dim == 0:
                image[lower:upper, :] = strip
            image = ((image / np.max(image)) * 255).astype(np.uint)
            return image

        def put_gaussian(mu, stdev, image_size):
            #stdev = torch.tensor([[0.05, 0.05]], device=device)
            point = multivariate_diag_gaussian(mu.view(1, -1), stdev.view(1, -1), image_size)
            image = point.cpu().numpy()
            image = ((image / np.max(image)) * 255).astype(np.uint)
            return image

        if label == 'all':
            player = put_strip(mu[0], stdev[0], 0.9, 0.05, image_size, dim=1)
            enemy = put_strip(mu[1], stdev[1], 0.3, 0.05, image_size, dim=1)
            ball = put_gaussian(mu[2:4], stdev[2:4], image_size)
            image = np.stack((player, enemy, ball.squeeze()))

        elif label == 'player':
            image = put_strip(mu, stdev, 0.9, 0.05, image_size, dim=1)

        elif label == 'enemy':
            image = put_strip(mu, stdev, 0.3, 0.05, image_size, dim=1)

        elif label == 'ball':
            image = put_gaussian(mu, stdev, image_size)
        else:
            raise Exception(f'label {label} not found')

        debug_image(image, block=False)


def train_predictor(mu_predictor, stdev_predictor, training, train_buff, test_buff, epochs, target_start, target_len, label, batch_size,
                    test_freq=50):
    train, test = SARDataset(train_buff), SARDataset(test_buff)
    pbar = Pbar(epochs=epochs, train_len=len(train), batch_size=batch_size, label=label)
    train = DataLoader(train, batch_size=batch_size, collate_fn=pad_collate, shuffle=True)
    test = DataLoader(test, batch_size=wandb.config.test_len, collate_fn=pad_collate, shuffle=True)

    if training == 'mu':
        optim = Adam(mu_predictor.parameters(), lr=1e-4)
    elif training == 'stdev':
        optim = Adam(mu_predictor.parameters(), lr=1e-4)
    else:
        optim = None

    eps = torch.finfo(next(iter(mu_predictor.parameters()))[0].data.dtype).eps
    train_cooldown = utils.Cooldown(secs=wandb.config.display_cooldown)

    def criterion(mu, stdev, target):
        dist = Normal(mu, stdev + eps)
        probs = dist.log_prob(target)
        return -torch.mean(probs * seqs.mask), dist

    def empirical_risk_criterion(estimate, target):
        return (((seqs.target - estimate) * seqs.mask) ** 2).mean()

    for epoch in range(epochs):

        for mb in train:
            seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, target_start, target_len).to(device)
            optim.zero_grad()
            with torch.no_grad():
                mu = mu_predictor(seqs.source, seqs.action)
            stdev = stdev_predictor(seqs.source, seqs.action)
            stdev = F.relu6(stdev)
            loss, dist = criterion(mu, stdev, seqs.target)
            loss.backward()

            optim.step()
            wandb.log({f'train_{label}_entropy': dist.entropy().mean(),
                       f'train_{label}_stdev': stdev.mean()})
            if training == 'mu':
                pbar.update_train_loss_and_checkpoint(loss, model=mu_predictor, epoch=epoch, optimizer=optim)
            else:
                pbar.update_train_loss_and_checkpoint(loss, model=stdev_predictor, epoch=epoch, optimizer=optim)

        if train_cooldown():
            with torch.no_grad():
                for mb in test:
                    seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, target_start, target_len).to(device)
                    mu = mu_predictor(seqs.source, seqs.action)
                    stdev = stdev_predictor(seqs.source, seqs.action)
                    stdev = F.relu6(stdev)
                    loss, dist = criterion(mu, stdev, seqs.target)

                    wandb.log({f'test_{label}_entropy': dist.entropy().mean()})
                    if training == 'mu':
                        pbar.update_test_loss_and_save_model(loss, model=mu_predictor)
                    else:
                        pbar.update_test_loss_and_save_model(loss, model=stdev_predictor)
                    display_predictions(mu[0], stdev[0], label)


class StaticStdev(nn.Module):
    def __init__(self, state_dims, action_dims, reward_dims, hidden_layers, output_dims, stdev=0.1):
        super().__init__()
        self.output_dims = output_dims
        self.stdev = stdev

    def forward(self, state, actions):
        return torch.full((state.size(0), state.size(1), self.output_dims),
                          fill_value=self.stdev, requires_grad=False, dtype=state.dtype, device=state.device)


def main():

    env = gym.make('PongNoFrameskip-v4')
    env = wrappers.NoopResetEnv(env, noop_max=30)
    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = AtariARIWrapper(env)
    env = wrappers.AtariAriVector(env)
    env = wrappers.FireResetEnv(env)

    # load dataset from file if already saved
    train_buff_path = Path(f'data/{wandb.config.mode}/train_buff.pkl')
    test_buff_path = Path(f'data/{wandb.config.mode}/test_buff.pkl')
    train_buff_path.parent.mkdir(parents=True, exist_ok=True)
    test_buff_path.parent.mkdir(parents=True, exist_ok=True)
    train_buff = load_or_generate(env, wandb.config.train_len, str(train_buff_path))
    test_buff = load_or_generate(env, wandb.config.test_len, str(test_buff_path))

    state_dims = 4
    action_dims = 6
    reward_dims = 1

    ensemble = {}
    #ensemble['all'] = SimpleNamespace(hidden=[512, 512, 512, 512], epochs=2000, target_start=0, target_len=4)
    #ensemble['player'] = SimpleNamespace(hidden=[512, 512, 512, 512], epochs=1000, target_start=0, target_len=1)
    #ensemble['enemy'] = SimpleNamespace(hidden=[512, 512, 512, 512], epochs=2000, target_start=1, target_len=1)
    ensemble['ball'] = SimpleNamespace(hidden=[512, 512, 512, 512], epochs=10000, target_start=2, target_len=2)

    # train_reward(buff, test_buff, epochs=10, test_freq=3)
    # train_done(buff, test_buff, epochs=10, test_freq=3)

    training = 'stdev'
    demo = False
    #load_dir = './wandb/dryrun-20200314_234615-0sxdtkko'
    load_dir = 'wandb/dryrun-20200315_183131-8ua370tg'


    if demo:

        test = SARDataset(test_buff)
        test = DataLoader(test, batch_size=wandb.config.test_len, collate_fn=pad_collate)
        predictor = {}

        with torch.no_grad():
            for label, args in ensemble.items():
                model = Causal(state_dims, action_dims, reward_dims, args.hidden, output_dims=args.target_len * 2).to(device)
                state_dict = Pbar.best_state_dict(load_dir, label)
                model.load_state_dict(state_dict)
                predictor[label] = model

            for mb in test:
                seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, args.target_start, args.target_len).to(device)
                estimate = predictor['all'](seqs.source, seqs.action)
                display_predictions(estimate[0], 'all', max_length=150)

            for mb in test:
                seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, args.target_start, args.target_len).to(device)
                estimate_player = predictor['player'](seqs.source, seqs.action)
                estimate_enemy = predictor['enemy'](seqs.source, seqs.action)
                estimate_ball = predictor['ball'](seqs.source, seqs.action)
                estimate = torch.cat((estimate_player, estimate_enemy, estimate_ball), dim=2)
                for trajectory in estimate:
                    display_predictions(trajectory, 'all', max_length=300)

    else:

        for label, args in ensemble.items():
            mu_predictor = models.causal.Causal(state_dims, action_dims, reward_dims, args.hidden, output_dims=args.target_len).to(device)
            if training == 'stdev':
                state_dict = Pbar.best_state_dict('wandb/dryrun-20200317_021617-hm3ki388', 'ball')
                mu_predictor.load_state_dict(state_dict)
            models.causal.init(mu_predictor, torch.nn.init.kaiming_normal)
            if training == 'mu':
                stdev_predictor = StaticStdev(state_dims, action_dims, reward_dims, args.hidden,
                                                output_dims=args.target_len, stdev=0.05).to(device)
            else:
                stdev_predictor = models.causal.Causal(state_dims, action_dims, reward_dims, args.hidden,
                                                output_dims=args.target_len).to(device)
            models.causal.init(stdev_predictor, torch.nn.init.kaiming_normal)
            train_predictor(mu_predictor=mu_predictor, stdev_predictor=stdev_predictor, training=training,
                            train_buff=train_buff, test_buff=test_buff, epochs=args.epochs,
                            target_start=args.target_start, target_len=args.target_len,
                            label=label, batch_size=wandb.config.predictor_batchsize,
                            test_freq=wandb.config.test_freq)




if __name__ == '__main__':
    defaults = dict(mode='test',
                    frame_op_len=8,
                    train_len=512,
                    test_len=16,
                    predictor_batchsize=32,
                    render=False,
                    rew_prefix_length=4,
                    rew_batchsize=32,
                    done_prefix_len=4,
                    done_batchsize=32,
                    test_freq=30,
                    display_cooldown=60)

    dev = dict(mode='dev',
               frame_op_len=8,
               train_len=2,
               test_len=2,
               predictor_batchsize=2,
               render=False,
               rew_prefix_length=4,
               rew_batchsize=256,
               done_prefix_len=4,
               done_batchsize=256,
               test_freq=300,
               display_cooldown=30)

    wandb.init(config=defaults)

    # config validations
    config = wandb.config
    assert config.predictor_batchsize <= config.train_len
    assert config.test_len >= 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()
