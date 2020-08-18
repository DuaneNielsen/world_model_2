import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torchdiffeq import odeint_adjoint as odeint
from wm2.env.LunarLander_v3 import LunarLander, heuristic
from wm2.env.gym_viz import VizWrapper
from wm2.data.datasets import Buffer, SARDataset
from wm2.data.utils import pad_collate_2
from wm2.utils import Cooldown
from gym.wrappers import TimeLimit
from wm2.models import vrnn


class Dynamics(nn.Module):

    def __init__(self, state_size=6, action_size=2, nhidden=512):
        super().__init__()
        self.vrnn = vrnn.VRNN(x_in_dim=8, x_out_dim=6, h_dim=512, z_dim=16, n_layers=1)

    def forward(self, state, action):
        return self.vrnn.forward(state, action)

    def sample(self):
        return self.vrnn.sample(16)


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
    fig = plt.figure(figsize=(12, 12))
    panels = [fig.add_subplot(6, 2, i, ) for i in range(1, 9)]
    state_panel_name = ['x', 'y', 'dx', 'dy', 'theta', 'omega', 'leg1', 'leg2', 's_power', 'm_power']
    action_panel_name = ['side thruster', 'main thruster']
    fig.canvas.set_window_title('Dynamics Training')
    cooldown = Cooldown(secs=5)

    """ data """
    buffer = generate_data(2)

    train = SARDataset(buffer, mask_f=None)
    train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

    """ setup ODE """
    dynamics = Dynamics().to(args.device)
    optim = Adam(dynamics.parameters(), lr=1e-4)  # Adam is much faster than RMSProp

    """ train """
    for i in range(10000):
        for trajectories in train:
            state, action = trajectories.state.to(device), trajectories.action.to(device)
            state, action = normalize(state), normalize(action)
            #loss_mask = trajectories.state.to(device).float()
            optim.zero_grad()
            kld_loss, nll_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std) = dynamics(state, action)
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
                for i, panel in enumerate(panels):
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
                # for i, panel in enumerate(panels[6:8]):
                #     panel.clear()
                #     panel.set_title(action_panel_name[i])
                #     panel.tick_params(
                #         axis='x',  # changes apply to the x-axis
                #         which='both',  # both major and minor ticks are affected
                #         bottom=False,  # ticks along the bottom edge are off
                #         top=False,  # ticks along the top edge are off
                #         labelbottom=False)
                #     panel.plot(t[:-1], _np(trajectories.action[:-1, 0, i]), label='control input')
                #     panel.legend()

                fig.canvas.draw()

    fig.savefig('{:03d}'.format(1100))
