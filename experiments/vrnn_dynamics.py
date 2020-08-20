import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import numpy as np
import argparse
from wm2.env.LunarLander_v3 import LunarLander, heuristic
from wm2.env.gym_viz import VizWrapper
from wm2.data.datasets import Buffer, SARDataset
from wm2.data.utils import pad_collate_2
from wm2.utils import Cooldown
from gym.wrappers import TimeLimit


import torch.utils
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

eps = torch.finfo(torch.float32).eps


class VRNN(nn.Module):
    def __init__(self, x_in_dim, x_out_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_in_dim = x_in_dim
        self.x_out_dim = x_out_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # feature-extracting transformations
        # x -> h
        self.phi_x = nn.Sequential(
            nn.Linear(x_in_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        # z -> h
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # encoder h -> z_mean, z_std
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # prior h -> z_mean, z_std
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoder h -> x_mean, x_std
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_out_dim),
            nn.Softplus())
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_out_dim),
            nn.Sigmoid())

        # recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x, c, loss_mask=None):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        eps = torch.finfo(torch.float32).eps

        x_in = torch.cat((x, c), dim=2)

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(x.device)

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x_in[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) + 0.01

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t) + 0.01

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t) + 0.01

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # computing losses
            #kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            # nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            if loss_mask is not None:
                kld_loss += torch.sum(self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t) * loss_mask[t])
                nll_loss += torch.sum(self._nll_bernoulli(dec_mean_t, x[t]) * loss_mask[t])
            else:
                nll_loss += torch.sum(self._nll_bernoulli(dec_mean_t, x[t]))
                kld_loss += torch.sum(self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t))
            # nll_loss += ((dec_mean_t - x[t]) ** 2).mean() / 2.0

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, nll_loss, \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std)

    @property
    def device(self):
        return self.dummy_param.device

    def sample(self, initial_x,  c):

        seq_len = c.shape[0]
        sample = torch.zeros(seq_len, self.x_out_dim, device=self.device)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim, device=self.device))

        h = self.phi_x(torch.cat((initial_x, c[0, 0]))).unsqueeze(0).unsqueeze(0)
        sample[0] = initial_x

        for t in range(1, seq_len-1):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(torch.cat((dec_mean_t, c[t, :]), dim=1))

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(mean.device)
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        return 0.5 * kld_element

    def _nll_bernoulli(self, theta, x):
        return - (x * torch.log(theta + eps) + (1 - x) * torch.log(1 - theta + eps))

    def _nll_gauss(self, mean, std, x):
        pass

def _np(x):
    return x.detach().cpu().numpy()


def action_to_vector(a):
    """[side_thruster, main_thruster]"""
    if a == 0:
        return np.array([0.0, 0.0])
    if a == 1:
        return np.array([-1.0, 0.0])
    if a == 2:
        return np.array([0.0, 1.0])
    if a == 3:
        return np.array([1.0, 0.0])


def demo_heuristic_lander(env, buffer, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    trajectory = buffer.next_traj()
    while True:
        a = heuristic(env, s)
        next_s, r, done, info = env.step(a)
        trajectory.append(s[0:6], action_to_vector(a), r, done, info)
        s = next_s

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done:
            trajectory.append(s[0:6], action_to_vector(0), r, done, info)
            break


def generate_data(episodes):
    """ data """
    env = LunarLander()
    env = TimeLimit(env, max_episode_steps=500)
    state_map = {
        'x': 0, 'y': 1, 'dx': 2, 'dy': 3, 'theta': 4, 'omega': 5
    }
    env = VizWrapper(env, state_map=state_map)

    buffer = Buffer(terminal_repeats=0)
    for i in range(episodes):
        demo_heuristic_lander(env, buffer)
    return buffer


def normalize(data):
    L, N, S = data.shape
    data = data.reshape(L*N, S)
    minimum = data.min(dim=0)[0]
    maximum = data.max(dim=0)[0]
    data = (data - minimum) / (maximum - minimum)
    return data.reshape(L, N, S)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('ODE demo')
    parser.add_argument('--method', type=str, choices=['euler', 'dopri5', 'adams'], default='rk4')
    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    """ figure """
    plt.ion()
    fig = plt.figure(figsize=(24, 16))
    subs = [fig.add_subplot(8, 3, i, ) for i in range(1, 25)]
    panel_index = [0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22, 2, 5, 8, 11, 14, 17, 20, 23]
    panels = []
    for i in panel_index:
        panels.append(subs[i])
    state_panel_name = ['x', 'y', 'dx', 'dy', 'theta', 'omega', 'side_thruster', 'main_thruster']
    fig.canvas.set_window_title('Dynamics Training')
    cooldown = Cooldown(secs=5)

    """ data """
    buffer = generate_data(16)

    train = SARDataset(buffer, mask_f=None)
    train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

    """ setup ODE """
    dynamics = VRNN(x_in_dim=8, x_out_dim=6, h_dim=512, z_dim=16, n_layers=1).to(args.device)
    optim = Adam(dynamics.parameters(), lr=1e-4)  # Adam is much faster than RMSProp

    """ train """
    for i in range(10000):
        for trajectories in train:
            state, action = trajectories.state.to(device), trajectories.action.to(device)
            state, action = normalize(state), normalize(action)
            loss_mask = trajectories.pad.to(device).float()
            optim.zero_grad()
            kld_loss, nll_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std) = dynamics(state, action, loss_mask)
            loss = (kld_loss + nll_loss)
            #loss = ((pred_y - trajectory) ** 2 * loss_mask).mean()
            loss.backward()
            optim.step()

            """ plot """
            print(loss.item())

            if cooldown():
                trajectory = torch.cat((state, action), dim=2)
                mean = torch.stack(all_dec_mean)
                std = torch.stack(all_dec_std)
                top = mean + std
                bottom = mean - std
                t = torch.linspace(0, len(trajectories.state) - 1, len(trajectories.state), device=args.device)
                t = _np(t)
                for i, panel in enumerate(panels[:8]):
                    panel.clear()
                    panel.set_title(state_panel_name[i])
                    panel.tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)
                    panel.plot(t, _np(trajectory[:, 0, i]), label='trajectory', color='blue')
                    if i < 6:
                        panel.plot(t, _np(mean[:, 0, i]), label='prediction', color='orange')
                        panel.fill_between(t, _np(bottom[:, 0, i]), _np(top[:, 0, i]), alpha=0.4, facecolor='orange')
                    panel.legend()

                def sample_plot(offset, initial_state, action, trajectory=None):
                    sample = dynamics.sample(initial_state, action)
                    for i, panel in enumerate(panels[offset:offset+8]):
                        panel.clear()
                        panel.set_title(state_panel_name[i])
                        panel.tick_params(
                            axis='x',  # changes apply to the x-axis
                            which='both',  # both major and minor ticks are affected
                            bottom=False,  # ticks along the bottom edge are off
                            top=False,  # ticks along the top edge are off
                            labelbottom=False)
                        if i < 6:
                            panel.plot(t, _np(sample[:, i]), color='grey')
                            if trajectory is not None:
                                panel.plot(t, _np(trajectory[:, 0, i]), label='trajectory', color='blue')
                        else:
                            panel.plot(t, _np(action[:, 0, i-6]), color='grey')

                # sample1 = dynamics.sample(action[:, 0:1, :])
                # sample2 = dynamics.sample(action[:, 1:2, :])

                sample_plot(offset=8, initial_state=state[0, 0], action=action[:, 0:1, :], trajectory=trajectory[:, 0:1])
                sample_plot(offset=16, initial_state=state[0, 1], action=action[:, 1:2, :], trajectory=trajectory[:, 1:2])

                # for i, panel in enumerate(panels[16:26]):
                #     panel.clear()
                #     panel.set_title(state_panel_name[i])
                #     panel.tick_params(
                #         axis='x',  # changes apply to the x-axis
                #         which='both',  # both major and minor ticks are affected
                #         bottom=False,  # ticks along the bottom edge are off
                #         top=False,  # ticks along the top edge are off
                #         labelbottom=False)
                #     if i < 6:
                #         panel.plot(t, _np(sample1[:, i]), label='sample 1', color='blue')
                #     else:
                #         panel.plot(t, _np(action[:, 0, i-6]), label='action 1', color='blue')
                #     panel.legend()

                fig.canvas.draw()

    fig.savefig('{:03d}'.format(1100))
