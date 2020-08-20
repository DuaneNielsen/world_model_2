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
from wm2.connectors.connector import ActionPipeline
from wm2.utils import Cooldown
from gym.wrappers import TimeLimit
from wm2.models.models import DreamTransitionModel
from wm2.models.vrnn import VRNN
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt



eps = torch.finfo(torch.float32).eps


class VRNNViz:
    def __init__(self, args):
        """ figure """
        plt.ion()
        self.fig = plt.figure(figsize=(24, 16))
        subs = [self.fig.add_subplot(8, 3, i, ) for i in range(1, 25)]
        panel_index = [0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22, 2, 5, 8, 11, 14, 17, 20, 23]
        self.panels = []
        for i in panel_index:
            self.panels.append(subs[i])
        self.state_panel_name = ['x', 'y', 'dx', 'dy', 'theta', 'omega', 'side_thruster', 'main_thruster']
        self.fig.canvas.set_window_title('Dynamics Training')
        self.cooldown = Cooldown(secs=5)
        self.imagine_cooldown = Cooldown(secs=5)

        self.args = args

    def sample_learn(self, state, action, mean, std):
        if self.cooldown():
            trajectory = torch.cat((state, action), dim=2)
            t_len = trajectory.shape[0]
            top = mean + std
            bottom = mean - std
            t = torch.linspace(0, t_len - 1, t_len, device=self.args.device)
            t = _np(t)

            for i, panel in enumerate(self.panels[:8]):
                panel.clear()
                panel.set_title(self.state_panel_name[i])
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

    def sample_imagine(self,  sample, trajectory):
        if self.imagine_cooldown():
            state = trajectory.state
            action = trajectory.action
            self.sample_plot(offset=8, sample=sample[:, 0:1, :], action=action[:, 0:1, :], trajectory=state[:, 0:1])
            self.sample_plot(offset=16, sample=sample[:, 1:2, :], action=action[:, 1:2, :], trajectory=state[:, 1:2])

    def sample_plot(self, offset, sample, action, trajectory=None):
        t_len = sample.shape[0]
        t = _np(torch.linspace(0, t_len - 1, t_len, device=self.args.device))
        for i, panel in enumerate(self.panels[offset:offset + 8]):
            panel.clear()
            panel.set_title(self.state_panel_name[i])
            panel.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)
            if i < 6:
                panel.plot(t, _np(sample[:, 0, i]), color='grey')
                if trajectory is not None:
                    panel.plot(t, _np(trajectory[torch.arange(t_len), 0, i]), label='trajectory', color='blue')
            else:
                panel.plot(t, _np(action[torch.arange(t_len), 0, i - 6]), color='grey')

    def draw(self):
        self.fig.canvas.draw()

    def save(self, name):
        self.fig.savefig(name)


class VRNNTransitionModel(DreamTransitionModel):
    def __init__(self, name, model):
        super().__init__(name, model)
        self.learn_samplers = {}
        self.imagine_samplers = {}

    def learn(self, args, buffer, optim):

        train = SARDataset(buffer, mask_f=None)
        train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

        for trajectories in train:
            state, action = trajectories.state.to(args.device), trajectories.action.to(args.device)
            state, action = normalize(state), normalize(action)
            loss_mask = trajectories.pad.to(device).float()
            optim.zero_grad()
            kld_loss, nll_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std) = self.model(state, action, loss_mask)
            loss = (kld_loss + nll_loss)
            loss.backward()
            optim.step()

            prd_mean = torch.stack(all_dec_mean)
            prd_std = torch.stack(all_dec_std)
            self.sample_learn(state, action, prd_mean, prd_std)

    def sample_learn(self, state, action, prediction_mean, prediction_variance=None):
        for key, sampler in self.learn_samplers.items():
            sampler.sample_learn(state, action , prediction_mean, prediction_variance)

    def imagine(self, args, trajectory, policy, action_pipeline):
        initial_state = trajectory.state[0].to(args.device)
        sample = self.model.sample(initial_state, policy, action_pipeline, args.horizon)
        self.sample_imagine(sample, trajectory)
        return sample

    def sample_imagine(self, sample, trajectory):
        for key, sampler in self.imagine_samplers.items():
            sampler.sample_imagine(sample, trajectory)


class DummyPolicy(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.step = 0
        self.actions = nn.Parameter(actions)

    def forward(self, state):
        action = self.actions[self.step]
        self.step += 1
        return action


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


def identity(arg):
    return arg


if __name__ == '__main__':

    parser = argparse.ArgumentParser('ODE demo')
    parser.add_argument('--method', type=str, choices=['euler', 'dopri5', 'adams'], default='rk4')
    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    """ data """
    buffer = generate_data(episodes=2)

    """ setup dynamics model """
    vrnn = VRNN(x_in_dim=8, x_out_dim=6, h_dim=512, z_dim=16, n_layers=1)
    dynamics = VRNNTransitionModel(name='vrnn_transition', model=vrnn).to(args.device)
    optim = Adam(dynamics.parameters(), lr=1e-4)  # Adam is much faster than RMSProp
    action_pipeline = ActionPipeline(identity, identity, identity, identity)

    """ viz """
    viz = VRNNViz(args)
    dynamics.learn_samplers['viz'] = viz
    dynamics.imagine_samplers['viz'] = viz

    """ train """
    for i in range(10000):
        dynamics.learn(args, buffer, optim)

        train = SARDataset(buffer, mask_f=None)
        train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
        trajectory = next(iter(train))
        policy = DummyPolicy(trajectory.action).to(args.device)
        dynamics.imagine(args, trajectory, policy, action_pipeline)
        viz.draw()
