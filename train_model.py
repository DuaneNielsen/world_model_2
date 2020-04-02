from time import sleep
from types import SimpleNamespace
import pickle
from pathlib import Path
from collections import deque

import gym
import numpy as np
from atariari.benchmark.wrapper import AtariARIWrapper
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.init
import wandb

from data.datasets import SARDataset, RewDataset, DoneDataset, gather_data
import wm2.env.wrappers as wrappers
from functional import compute_covar
from keypoints.utils import UniImageViewer
from utils import Pbar
from viz import debug_image, put_strip, put_gaussian, display_predictions
from data.utils import pad_collate, autoregress, chomp_and_pad
from wm2.models.causal import Causal
import utils
import models.causal


viewer = UniImageViewer()


def reward_value_estimator(state_dict=None):
    model = models.causal.RewardNet(4, [512, 512], 1,  scale=1.2)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def reward_class_estimator(state_dict=None):
    model = models.causal.Encoder(4, 0, 0, [512, 512], 3)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def done_model(state_dict=None):
    model = models.causal.DoneNet(4, [512, 512])
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def train_reward_class(buff, test_buff, epochs, test_freq=2):
    train = RewDataset(buff, prefix_len=wandb.config.rew_prefix_length, prefix_mode='stack')
    pbar = Pbar(epochs, len(train), wandb.config.rew_batchsize, "reward_class")
    weights = train.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    train = DataLoader(train, batch_size=wandb.config.rew_batchsize, sampler=sampler, drop_last=True)

    test = RewDataset(test_buff, prefix_len=wandb.config.rew_prefix_length, prefix_mode='stack')
    weights = test.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    test = DataLoader(test, batch_size=wandb.config.rew_batchsize, sampler=sampler, drop_last=True)

    model = reward_class_estimator()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for source, reward, reward_class in train:
            optim.zero_grad()
            estimate = model(source)
            loss = nn.functional.cross_entropy(estimate[:, -1, :].squeeze(), reward_class)
            loss.backward()
            optim.step()
            wandb.log({'reward_loss': loss.item()})
            pbar.update_train_loss_and_checkpoint(loss, models={'reward': model}, optimizer=optim, epoch=epoch)

        if epoch % test_freq == 0:
            confusion = torch.zeros([3, 3], dtype=torch.long, device=wandb.config.device)
            with torch.no_grad():
                for source, reward, reward_class in test:
                    estimate = model(source)
                    loss = nn.functional.cross_entropy(estimate[:, -1, :].squeeze(), reward_class)
                    wandb.log({'reward_loss': loss.item()})
                    pbar.update_test_loss_and_save_model(loss, models={'reward': model})
                    estimate = torch.argmax(estimate[:, -1, :], dim=1)
                    for e, l in zip(estimate, reward_class):
                        confusion[e, l] += 1
                print('')
                print(confusion)
    pbar.close()


def train_reward_value(buff, test_buff, epochs, test_freq=2):
    train = RewDataset(buff, prefix_len=wandb.config.rew_prefix_length, prefix_mode='stack')
    pbar = Pbar(epochs, len(train), wandb.config.rew_batchsize, "reward_value")
    weights = train.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    train = DataLoader(train, batch_size=wandb.config.rew_batchsize, sampler=sampler, drop_last=True)

    test = RewDataset(test_buff, prefix_len=wandb.config.rew_prefix_length, prefix_mode='stack')
    weights = test.weights()
    sampler = WeightedRandomSampler(weights, len(weights))
    test = DataLoader(test, batch_size=wandb.config.rew_batchsize, sampler=sampler, drop_last=True)

    model = reward_value_estimator()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for source, reward, reward_class in train:
            optim.zero_grad()
            estimate = model(source)
            loss = ((reward.squeeze() - estimate[:, 3, 0]) ** 2).mean()
            loss.backward()
            optim.step()
            wandb.log({'reward_loss': loss.item()})
            pbar.update_train_loss_and_checkpoint(loss, models={'reward': model}, optimizer=optim, epoch=epoch)

        if epoch % test_freq == 0:
            with torch.no_grad():
                for source, reward, reward_class in test:
                    estimate = model(source)
                    loss = ((reward.squeeze() - estimate[:, 3, 0]) ** 2).mean()
                    wandb.log({'reward_loss': loss.item()})
                    pbar.update_test_loss_and_save_model(loss, models={'reward': model})
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

    model = done_model()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for source, target in train:
            optim.zero_grad()
            estimate = model(source)
            loss = criterion(estimate[:, -1, 0].squeeze(), target.squeeze())
            loss.backward()
            optim.step()
            wandb.log({'done_loss': loss.item()})
            pbar.update_train_loss_and_checkpoint(loss, models={'done': model}, optimizer=optim, epoch=epoch)

        if epoch % test_freq == 0:
            with torch.no_grad():
                confusion = torch.zeros(2, 2)
                for source, target in test:
                    estimate = model(source)
                    loss = criterion(estimate[:, -1, 0].squeeze(), target.squeeze())
                    wandb.log({'done_loss': loss.item()})
                    pbar.update_test_loss_and_save_model(loss, models={'done': model})
                    estimate = torch.round(estimate[:, -1, 0].squeeze()).long()
                    for e, l in zip(estimate, torch.round(target).long()):
                        confusion[e, l] += 1
                print('')
                print(confusion)
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


def make_future_seq(inp, future_timesteps=3):
    inp_batch = []
    prev_inp = inp.clone()

    for _ in range(future_timesteps):
        inp_batch += [prev_inp]
        prev_inp = chomp_and_pad(inp, dim=1)

    return torch.cat(inp_batch)


def train_predictor(mu_encoder, mu_decoder, train_buff, test_buff, epochs, target_start, target_len, label, batch_size,
                    test_freq=50):
    train, test = SARDataset(train_buff), SARDataset(test_buff)
    pbar = Pbar(epochs=epochs, train_len=len(train), batch_size=1, label=label)
    train = DataLoader(train, batch_size=1, collate_fn=pad_collate, shuffle=True)
    test = DataLoader(test, batch_size=1, collate_fn=pad_collate, shuffle=True)

    optim = Adam(mu_encoder.parameters(), lr=1e-4)

    eps = torch.finfo(next(iter(mu_encoder.parameters()))[0].data.dtype).eps
    train_cooldown = utils.Cooldown(secs=wandb.config.display_cooldown)

    for epoch in range(epochs):

        for mb in train:
            seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, target_start, target_len).to(device)
            optim.zero_grad()
            inp = torch.cat((seqs.source, seqs.action), dim=2)
            hidden = mu_encoder(inp)

            h = hidden.repeat(mu_decoder.layers, 1, 1).contiguous()
            c = hidden.clone().repeat(mu_decoder.layers, 1, 1).contiguous()

            inp_future = make_future_seq(inp)
            tar_future = make_future_seq(seqs.target)

            mu, (h, c) = mu_decoder(inp_future, (h, c))

            loss = ((mu - tar_future) ** 2).mean()
            loss.backward()
            optim.step()

            pbar.update_train_loss_and_checkpoint(loss, models={'encoder': mu_encoder,
                                                                'decoder': mu_decoder}, epoch=epoch, optimizer=optim)

            if train_cooldown():
                with torch.no_grad():
                    for i, mb in enumerate(test):
                        seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, target_start, target_len).to(device)
                        inp = torch.cat((seqs.source, seqs.action), dim=2)
                        hidden = mu_encoder(inp)

                        inp_future = make_future_seq(inp)
                        tar_future = make_future_seq(seqs.target)

                        samples = []
                        for _ in range(10):
                            h = hidden.repeat(mu_decoder.layers, 1, 1).contiguous()
                            c = hidden.clone().repeat(mu_decoder.layers, 1, 1).contiguous()
                            mu, (h, c) = mu_decoder(inp_future, (h, c))
                            samples += [mu]

                        samples = torch.stack(samples)

                        mu = samples.mean(dim=0)

                        loss = ((mu - tar_future) ** 2).mean()

                        covar = compute_covar(samples, mu)
                        # wandb.log({f'test_{label}_entropy': dist.entropy().mean()})
                        pbar.update_test_loss_and_save_model(loss, models={'encoder': mu_encoder,
                                                                           'decoder': mu_decoder})
                        if i == 0:
                            display_predictions(mu, trajectory_covar=covar, label=label)


class Extrapolation:
    def __init__(self, config, load_dir):
        self.config = config
        self.mu = {k: [] for k in config}
        self.covar = {k: [] for k in config}
        self.mu_2 = {k: [] for k in config}
        self.covar_2 = {k: [] for k in config}
        self.samples = {k: [] for k in config}
        self.seqs = None
        self.encoders = {}
        self.decoders = {}
        self.done = None
        self.reward_class = None
        self.reward_value = None
        self._load(config, load_dir)

    def _load(self, config, load_dir):
        for label, args in config.items():
            encoder = models.causal.Encoder(args.state_dims, args.action_dims, args.reward_dims, args.hidden,
                                               output_dims=args.hidden_state_dims).to(device)
            decoder = models.causal.Decoder(args.state_dims, args.action_dims, args.reward_dims, args.hidden_state_dims,
                                               args.target_len).to(device)
            state_dicts = Pbar.best_state_dict(load_dir, label)
            encoder.load_state_dict(state_dicts['encoder'])
            decoder.load_state_dict(state_dicts['decoder'])
            self.encoders[label] = encoder
            self.decoders[label] = decoder

        self.reward_value = reward_value_estimator(Pbar.best_state_dict(load_dir, 'reward_value')['reward'])

        self.reward_class = reward_class_estimator(
            Pbar.best_state_dict(load_dir, 'reward_class')['reward'])

    def extrapolate(self, trajectories, extrap_len=3):
        with torch.no_grad():
            self.seqs = autoregress(trajectories.state, trajectories.action, trajectories.reward,
                                    trajectories.mask).to(device)
            inp = torch.cat((self.seqs.source, self.seqs.action), dim=2)
            #seq = torch.empty_like(self.seqs.source)
            h = {}
            c = {}
            out = {}

            for _ in range(10):
                for k, args in self.config.items():
                    hidden = self.encoders[k](inp)
                    h[k] = hidden.repeat(self.decoders[k].layers, 1, 1).contiguous()
                    c[k] = hidden.clone().repeat(self.decoders[k].layers, 1, 1).contiguous()
                for t in range(extrap_len):
                    for k, args in self.config.items():
                        out[k], (h[k], c[k]) = self.decoders[k](inp, (h[k], c[k]))
                    state = torch.cat((out['player'], out['enemy'], out['ball']), dim=2)
                    reward_class = self.reward_class(state)
                    action_shape = list(inp.shape)
                    action_shape[2] = 6
                    action = torch.zeros(*action_shape, device=state.device)
                    action[0, torch.arange(action.size(1)), torch.randint(0, 5, (action.size(1),))] = 1.0
                    inp = torch.cat((state, reward, action), dim=2)
                    self.samples[k] += [inp]

                # seq[:, :, args.target_start:args.target_start + args.target_len] = samples[k]
                self.samples[k] = torch.stack(self.samples[k])
                self.mu[k] = self.samples[k].mean(dim=0)
                self.covar[k] = compute_covar(self.samples[k], self.mu[k])

            # for k, v in ensemble.items():
            #     samples[k] = simple_sample(1, self.mu[k], self.covar[k])

    def play(self, trajectory_id, max_length=100, fps=6, image_size=(240 * 4, 160 * 4)):
        mask_len = self.seqs.mask[trajectory_id].sum()
        length = min(mask_len, max_length)
        for step in range(length):
            frame = []
            for k in self.config:
                if self.config[k].target_len == 1:
                    config = self.config[k]
                    channel = put_strip(image_size, config.x, config.thickness, dim=config.dim,
                                        mu=self.mu[k][trajectory_id, step],
                                        covar=self.covar[k][trajectory_id, step])
                    frame.append(channel)
                elif self.config[k].target_len == 2:
                    channel = put_gaussian(image_size, mu=self.mu[k][trajectory_id, step],
                                           covar=self.covar[k][trajectory_id, step])
                    frame.append(channel.squeeze())

            image = np.stack(frame)
            debug_image(image, block=False)
            sleep(1 / fps)


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

    ensemble = {}
    ensemble['player'] = SimpleNamespace(state_dims=4, action_dims=6, reward_dims=1, hidden=[512, 512, 512, 512], hidden_state_dims=512, epochs=100,
                                         target_start=0,
                                         target_len=1,
                                         x=0.9, thickness=0.05, dim=1)
    ensemble['enemy'] = SimpleNamespace(state_dims=4, action_dims=6, reward_dims=1, hidden=[512, 512, 512, 512], hidden_state_dims=512, epochs=200, target_start=1,
                                        target_len=1,
                                        x=0.3, thickness=0.05, dim=1)
    ensemble['ball'] = SimpleNamespace(state_dims=4, action_dims=6, reward_dims=1, hidden=[512, 512, 512, 512], hidden_state_dims=512, epochs=500, target_start=2,
                                       target_len=2)


    # wandb.config.demo = False
    load_dir = 'wandb/dryrun-20200401_061120-1i0saqke'
    load_reward_done_dir = ''

    if not wandb.config.demo:
        train_reward_class(train_buff, test_buff, epochs=50, test_freq=3)
        train_reward_value(train_buff, test_buff, epochs=50, test_freq=3)
        train_done(train_buff, test_buff, epochs=10, test_freq=3)

        for label, args in ensemble.items():
            mu_encoder = models.causal.Encoder(args.state_dims, args.action_dims, args.reward_dims, args.hidden,
                                               output_dims=args.hidden_state_dims).to(device)
            mu_decoder = models.causal.Decoder(args.state_dims, args.action_dims, args.reward_dims, args.hidden_state_dims,
                                               args.target_len).to(device)
            models.causal.init(mu_encoder, torch.nn.init.kaiming_normal)
            train_predictor(mu_encoder=mu_encoder, mu_decoder=mu_decoder,
                            train_buff=train_buff, test_buff=test_buff, epochs=args.epochs,
                            target_start=args.target_start, target_len=args.target_len,
                            label=label, batch_size=wandb.config.predictor_batchsize,
                            test_freq=wandb.config.test_freq)

    else:

        test = SARDataset(test_buff)
        test = DataLoader(test, batch_size=1, collate_fn=pad_collate)
        ex = Extrapolation(ensemble, load_dir)

        for mb in test:
            ex.extrapolate(mb)
            ex.play(trajectory_id=0, max_length=1000)
            ex.play(trajectory_id=1, max_length=1000)


if __name__ == '__main__':
    test = dict(mode='test',
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
                display_cooldown=60,
                device='cuda:0',
                demo=False)

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
               display_cooldown=30,
               device='cuda:1',
               demo=True)

    wandb.init(config=dev)

    # config validations
    config = wandb.config
    assert config.predictor_batchsize <= config.train_len
    assert config.test_len >= 2

    device = wandb.config.device

    main()
