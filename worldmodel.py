from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import numpy as np
import wandb

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

        if pos >= 1.0:
            return pos, 1.0, True
        elif self.step_count > 10:
            return pos, 0.0, True
        else:
            return pos, 0.0, False


def prepro(state):
    return torch.tensor(state).unsqueeze(0)


def gather_seed_episodes(seed_episodes):
    env = LinEnv()
    buff = Buffer()
    dist = TanhTransformedGaussian(0.0, 0.5)

    for episode in range(seed_episodes):
        state, reward, done = env.reset(), 0.0, False
        action = dist.sample()
        buff.append(episode, state, action.cpu().numpy(), reward, done, None)
        while not done:
            state, reward, done = env.step(action.squeeze(0).cpu().numpy())
            action = dist.sample()
            buff.append(episode, state, action.cpu().numpy(), reward, done, None)

    return buff


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(1, 1)
        self.scale = nn.Linear(1, 1)

    def forward(self, state):
        mu, scale = self.mu(state), self.scale(state)
        return TanhTransformedGaussian(mu, scale)


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
    train_buff = gather_seed_episodes(args.seed_episodes)
    test_buff = gather_seed_episodes(args.seed_episodes)

    # policies here
    policy = Policy()
    policy_optim = Adam(policy.parameters(), lr=1e-4)

    # transition model
    T = nn.LSTM(input_size=2, hidden_size=1, num_layers=1)
    T_optim = Adam(T.parameters(), lr=1e-4)

    # reward model
    R = nn.Linear(1, 1)
    R_optim = Adam(R.parameters(), lr=1e-4)

    # terminal state model
    D = nn.Linear(1, 1)
    D_optim = Adam(D.parameters(), lr=1e-4)

    converged = False

    while not converged:
        for c in range(args.collect_interval):

            # Dynamics learning
            train, test = SARNextDataset(train_buff, mask_f=None), SARNextDataset(test_buff, mask_f=None)
            train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            pbar = Pbar(items_to_process=args.trajectories_per_pass, train_len=len(train), batch_size=args.batch_size, label='transition')
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

            # Behaviour learning
            

            pbar.close()

        converged = True




if __name__ == '__main__':
    args = {'seed_episodes': 10,
            'collect_interval': 30,
            'batch_size': 1,
            'trajectories_per_pass': 10,
            'device': 'cuda:0'
            }

    wandb.init(config=args)
    args = SimpleNamespace(**args)

    main(args)