from time import sleep
from types import SimpleNamespace
import pickle
from pathlib import Path

import gym
import numpy as np
from atariari.benchmark.wrapper import AtariARIWrapper
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
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
            h = mu_encoder(inp)

            h = h.contiguous()
            c = h.clone().contiguous()

            inp_batch = []
            tar_batch = []
            prev_inp = inp.clone()
            prev_tar = seqs.target

            for _ in range(2):
                inp_batch += [prev_inp]
                tar_batch += [prev_tar]

                prev_inp = chomp_and_pad(inp, dim=1)
                prev_tar = chomp_and_pad(seqs.target, dim=1)

            inp_future = torch.cat(inp_batch)
            tar_future = torch.cat(tar_batch)

            mu, (h, c) = mu_decoder(inp_future, (h, c))

            loss = ((mu - tar_future) ** 2).mean()
            loss.backward()
            optim.step()

            pbar.update_train_loss_and_checkpoint(loss, model=mu_encoder, epoch=epoch, optimizer=optim)

        if train_cooldown():
            with torch.no_grad():
                for mb in test:
                    seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, target_start, target_len).to(device)
                    inp = torch.cat((seqs.source, seqs.action), dim=2)
                    hidden_encodings = []
                    for _ in range(10):
                        hidden_encodings += [mu_encoder(inp)]
                    episodes = []
                    for e in range(len(inp)):
                        samples = []
                        for j in range(len(hidden_encodings)):
                            i = inp[e]
                            h = hidden_encodings[j][e]
                            h, c = h.unsqueeze(0).contiguous(), h.clone().unsqueeze(0).contiguous()
                            mu, (h, c) = mu_decoder(i.unsqueeze(0), (h, c))
                            samples += [mu]
                        episodes.append(torch.stack(samples))

                    episodes = torch.cat(episodes, dim=1)
                    mu = episodes.mean(dim=0)

                    loss = ((mu - seqs.target)**2).mean()

                    covar = compute_covar(episodes, mu)
                    # wandb.log({f'test_{label}_entropy': dist.entropy().mean()})
                    pbar.update_test_loss_and_save_model(loss, model=mu_encoder)
                    display_predictions(mu[0], trajectory_covar=covar[0], label=label)


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
    # ensemble['all'] = SimpleNamespace(hidden=[512, 512, 512, 512], epochs=2000, target_start=0, target_len=4)
    # ensemble['player'] = SimpleNamespace(hidden=[512, 512, 512, 512], hidden_state_dims=512, epochs=1000, target_start=0,
    #                                      target_len=1,
    #                                      x=0.9, thickness=0.05, dim=1)
    # ensemble['enemy'] = SimpleNamespace(hidden=[512, 512, 512, 512], hidden_state_dims=512, epochs=2000, target_start=1, target_len=1,
    #                                    x=0.3, thickness=0.05, dim=1)
    ensemble['ball'] = SimpleNamespace(hidden=[512, 512, 512, 512], hidden_state_dims=512, epochs=5000, target_start=2, target_len=2)

    # train_reward(train_buff, test_buff, epochs=10, test_freq=3)
    # train_done(buff, test_buff, epochs=10, test_freq=3)

    training = 'stdev'
    #wandb.config.demo = False
    # load_dir = './wandb/dryrun-20200314_234615-0sxdtkko'
    load_dir = 'wandb/run-20200319_020617-xbjdabd8'

    if wandb.config.demo:

        test = SARDataset(test_buff)
        test = DataLoader(test, batch_size=wandb.config.test_len, collate_fn=pad_collate)
        predictor = {}

        class Extrapolation:
            def __init__(self, config, load_dir):
                self.config = config
                self.ensemble = {}
                self._load(config, load_dir)
                self.mu = {k: [] for k in ensemble}
                self.covar = {k: [] for k in ensemble}
                self.mu_2 = {k: [] for k in ensemble}
                self.covar_2 = {k: [] for k in ensemble}
                self.seqs = None

            def _load(self, config, load_dir):
                for label, args in config.items():
                    model = Causal(state_dims, action_dims, reward_dims, args.hidden, output_dims=args.target_len).to(
                        device)
                    state_dict = Pbar.best_state_dict(load_dir, label)
                    model.load_state_dict(state_dict)
                    self.ensemble[label] = model

            def extrapolate(self, trajectories):
                with torch.no_grad():
                    self.seqs = autoregress(trajectories.state, trajectories.action, trajectories.reward,
                                            trajectories.mask).to(device)
                    samples = {k: [] for k in ensemble}
                    seq = torch.empty_like(self.seqs.source)

                    for k, args in self.config.items():
                        for _ in range(32):
                            sample = self.ensemble[k](self.seqs.source, self.seqs.action)
                            samples[k] += [sample]

                        #seq[:, :, args.target_start:args.target_start + args.target_len] = samples[k]
                        samples[k] = torch.stack(samples[k])
                        self.mu[k] = samples[k].mean(dim=0)
                        self.covar[k] = compute_covar(samples[k], self.mu[k])

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

        ex = Extrapolation(ensemble, load_dir)

        for mb in test:
            ex.extrapolate(mb)
            ex.play(trajectory_id=0, max_length=1000)
            ex.play(trajectory_id=1, max_length=1000)

        # with torch.no_grad():
        # for label, args in ensemble.items():
        #     model = Causal(state_dims, action_dims, reward_dims, args.hidden, output_dims=args.target_len).to(device)
        #     state_dict = Pbar.best_state_dict(load_dir, label)
        #     model.load_state_dict(state_dict)
        #     predictor[label] = model

        # for mb in test:
        #     seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, args.target_start, args.target_len).to(device)
        #     estimate = predictor['all'](seqs.source, seqs.action)
        #     display_predictions(estimate[0], 'all', max_length=150)

        # for mb in test:
        #     seqs = autoregress(mb.state, mb.action, mb.reward, mb.mask, args.target_start, args.target_len).to(device)
        #
        #     samples = {k: [] for k in ensemble}
        #     mu = {k: [] for k in ensemble}
        #     covar = {k: [] for k in ensemble}
        #
        #     for k, v in ensemble.items():
        #         for _ in range(10):
        #             estimate = predictor[k](seqs.source, seqs.action)
        #             samples[k] += [estimate]
        #
        #         samples[k] = torch.stack(samples[k])
        #         mu[k] = samples[k].mean(dim=0)
        #         covar[k] = compute_covar(samples[k], mu[k])
        #
        #     for k in ensemble:
        #         for m, c in zip(mu[k], covar[k]):
        #             display_predictions(trajectory_mu=m, trajectory_covar=c, label=k, max_length=300)

    else:

        for label, args in ensemble.items():
            mu_encoder = models.causal.Encoder(state_dims, action_dims, reward_dims, args.hidden,
                                               output_dims=args.hidden_state_dims).to(device)
            mu_decoder = models.causal.Decoder(state_dims, action_dims, reward_dims, args.hidden_state_dims,
                                               args.target_len).to(device)
            models.causal.init(mu_encoder, torch.nn.init.kaiming_normal)
            train_predictor(mu_encoder=mu_encoder, mu_decoder=mu_decoder,
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
               demo=False)

    wandb.init(config=dev)

    # config validations
    config = wandb.config
    assert config.predictor_batchsize <= config.train_len
    assert config.test_len >= 2

    device = wandb.config.device

    main()
