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


class Dynamics(nn.Module):

    def __init__(self, state_size=6, action_size=2, nhidden=512):
        super(Dynamics, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(state_size + action_size, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, state_size, bias=False)
        torch.nn.init.zeros_(self.fc3.weight.data)
        self.nfe = 0

    def forward(self, t, state):
        index = torch.floor(t).long()
        index = index.clamp(0, self.h.shape[0] - 1)
        actions = self.h[index]
        controlled = torch.cat([state, actions], dim=1)
        self.nfe += 1
        hidden = self.fc1(controlled)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.elu(hidden)
        dstate = self.fc3(hidden)
        return dstate


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
    fig.canvas.set_window_title('Learning from FC network only')
    cooldown = Cooldown(secs=5)

    """ data """
    buffer = generate_data(32)
    train = SARDataset(buffer, mask_f=None)
    train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

    """ setup ODE """
    func = Dynamics().to(args.device)
    optim = Adam(func.parameters(), lr=1e-4)  # Adam is much faster than RMSProp

    """ train """
    for i in range(10000):
        for trajectories in train:
            func.h = trajectories.action.to(args.device)
            t = torch.linspace(0, len(trajectories.state) - 2, len(trajectories.state) - 1, device=args.device)
            y0, trajectory, loss_mask = trajectories.state[0].to(args.device), trajectories.state[1:].to(args.device), \
                                        trajectories.pad[1:].to(args.device)
            optim.zero_grad()
            pred_y = odeint(func, y0, t, method=args.method)
            loss = ((pred_y - trajectory) ** 2 * loss_mask).mean()
            loss.backward()
            optim.step()

            """ plot """
            print(loss.item())

            if cooldown():
                func.h = trajectories.action.to(args.device)
                t = torch.linspace(0, len(trajectories.state) - 2, len(trajectories.state) - 1, device=args.device)
                y0, trajectory = trajectories.state[0].to(args.device), trajectories.state[1:].to(args.device)
                pred_y = odeint(func, y0, t, method=args.method)
                for i, panel in enumerate(panels[0:6]):
                    panel.clear()
                    panel.set_title(state_panel_name[i])
                    panel.tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)
                    panel.plot(_np(t), _np(trajectory[:, 0, i]), label='trajectory')
                    panel.plot(_np(t), _np(pred_y[:, 0, i]), label='prediction')
                    panel.legend()
                for i, panel in enumerate(panels[6:8]):
                    panel.clear()
                    panel.set_title(action_panel_name[i])
                    panel.tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)
                    panel.plot(_np(t), _np(trajectories.action[:-1, 0, i]), label='control input')
                    panel.legend()

                fig.canvas.draw()

    fig.savefig('{:03d}'.format(1100))
