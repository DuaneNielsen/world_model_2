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
import gym
from matplotlib.animation import FuncAnimation
from wm2.models.mnn import make_layers
from wm2.models.layerbuilder import LayerMetaData
from torchvision.transforms import Normalize, Compose, Grayscale, ToPILImage, ToTensor, Resize


"""  This doesn't work, better to do something like

https://arxiv.org/abs/1809.03137 Tracking by Animation: Unsupervised Learning of Multi-Object Attentive Trackers

OR

https://mjlm.github.io/video_structure/ Unsupervised Learning of Object Structure and Dynamics from Videos

"""

class ImageDynamics(nn.Module):
    def __init__(self):
        super().__init__()

        meta = LayerMetaData(input_shape=(2, 128, 128))
        self.encoder_core, meta = make_layers(['C:1', 64, 'M', 64, 'M', 128, 'M', 128, 'M', 256, 'M', 256, 'M'],
                                              type='resnet', meta=meta, nonlinearity=nn.LeakyReLU)

        self.decoder_core, meta = make_layers([256, 'U', 256, 'U', 128, 'U', 128, 'U', 64, 'U', 64, 'U', 'C:1'],
                                              type='resnet', meta=meta, nonlinearity=nn.LeakyReLU)

        latent_size = 256 * 2 * 2
        self.z = nn.Sequential(nn.Linear(latent_size, latent_size), nn.LeakyReLU(), nn.Linear(latent_size, 12), nn.LeakyReLU())
        self.z_dec = nn.Sequential(nn.Linear(12, latent_size), nn.LeakyReLU(), nn.Linear(latent_size, latent_size), nn.LeakyReLU())

        # meta = LayerMetaData(input_shape=(6, 208, 160))
        # self.d1_image, meta = make_layers(['C:3', 64, 128, 128, 64, 'C:3'],
        #                                  type='resnet', meta=meta, nonlinearity=nn.Tanh)

    def forward(self, t, state):
        d_image = state[:, 1:2, :, :]
        z2_image = self.encoder_core(d_image)
        # N, C, H, W = z2_image.shape
        # z2 = self.z(z2_image.reshape(N, -1))
        # z2_image = self.z_dec(z2).reshape(N, C, H, W)
        d2_image = self.decoder_core(z2_image)
        state = torch.cat((d_image, d2_image), dim=1)
        return state


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


def to_tensor(state):
    state = torch.from_numpy(state).float().to(device)
    return state.permute(2, 0, 1) / 255.0


def to_numpy(image):
    image = image.permute(1, 2, 0) * 255
    state = torch.floor(image).byte()
    return state.detach().cpu().numpy()


def to_numpy_greyscale(image):
    # image = image.permute(1, 2, 0) * 255
    state = torch.floor(image).byte()
    return state.detach().cpu().numpy()


class Panel:
    def __init__(self):
        plt.ion()
        self.fig, (self.input_image, self.dx_subp) = plt.subplots(1, 2)
        self.fig.canvas.set_window_title('Visual Dynamics')
        self.im = None
        self.di_image = None

    def reset(self, state, ode_func):
        self.im = self.input_image.imshow(state)
        state = to_tensor(state).unsqueeze(0)
        di = ode_func(state)
        di = to_numpy(di.squeeze(0))
        self.di_image = self.dx_subp.imshow(di)

    def step(self, state, ode_func):
        self.im.set_data(state)
        state = to_tensor(state).unsqueeze(0)
        di = ode_func(state)
        di = to_numpy(di.squeeze(0))
        self.di_image.set_data(di)
        self.fig.canvas.draw()


class PredictionPanel:
    def __init__(self):
        plt.ion()
        self.fig, self.subs = plt.subplots(2, 2)
        self.im = [[None, None], [None, None]]
        self.fig.canvas.set_window_title('Prediction Dynamics')

    def update_image(self, i, j, image):
        if len(image.shape) == 3:
            image = to_numpy(image)
        else:
            image = to_numpy_greyscale(image)
        if self.im[i][j] is None:
            self.im[i][j] = self.subs[i][j].imshow(image)
        else:
            self.im[i][j].set_data(image)

    def update(self, *args):
        import math
        for i, arg in enumerate(args):
            row = math.floor(i / 2)
            col = i % 2
            self.update_image(row, col, arg)
        plt.pause(1 / 6)
        self.fig.canvas.draw()


def generate_data(episodes):
    """ data """
    env = gym.make('Pong-v4')
    transform = Compose([ToPILImage(), Grayscale(), Resize([128, 128]), ToTensor(), Normalize(0.5, 0.5)])

    # buffer = Buffer(terminal_repeats=0)
    for i in range(episodes):
        states = []
        state, reward, done, info = env.reset(), 0.0, False, {}
        state = state[34:180]
        action = env.action_space.sample()
        states += [transform(state)]
        # viz.reset(state, ode_func)
        while not done:
            state, reward, done, info = env.step(action)
            state = state[34:180]
            action = env.action_space.sample()
            states += [transform(state)]
            print(len(states))
            # viz.step(state, ode_func)

        return torch.stack(states)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('ODE demo')
    parser.add_argument('--method', type=str, choices=['euler', 'dopri5', 'adams', 'rk4'], default='rk4')
    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    """ figure """
    # viz = Panel()
    predict = PredictionPanel()
    cooldown = Cooldown(secs=5)

    """ setup ODE """
    func = ImageDynamics().to(args.device)
    optim = Adam(func.parameters(), lr=1e-3)  # Adam is much faster than RMSProp

    """ data """
    states = generate_data(1)
    states = states[32:64]
    # train = SARDataset(buffer, mask_f=None)
    # train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

    """ compute new trajectory with d_image """
    buffer = []
    prev = states[0]
    for state in states[1:]:
        d_state = state - prev
        buffer += [torch.cat((state, d_state))]
        prev = state.clone()
    states = torch.stack(buffer)

    y0, trajectory = [], []
    for b in range(2):
        y0 += [states[b]]
        trajectory += [states[b + 1:b + 18]]
    y0, trajectory = torch.stack(y0).to(args.device), torch.stack(trajectory, dim=1).to(args.device)
    timesteps = trajectory.shape[0]
    t = torch.linspace(0, timesteps - 1, timesteps, device=args.device)

    for i in range(1000000000):
        optim.zero_grad()
        pred_y = odeint(func, y0, t, method=args.method)
        loss = ((pred_y - trajectory) ** 2).mean()
        loss.backward()
        optim.step()

        print(loss.item())
        if i % 10 == 0:
            for orig, pred in zip(trajectory[:, 0], pred_y[:, 0]):
                # predict.update(orig[0:3], orig[3:6], pred[0:3], pred[3:6])
                predict.update(orig[0], orig[1], pred[0], pred[1])

    # """ train """
    # for i in range(10000):
    #     for trajectories in train:
    #         func.h = trajectories.action.to(args.device)
    #         t = torch.linspace(0, len(trajectories.state) - 2, len(trajectories.state) - 1, device=args.device)
    #         y0, trajectory, loss_mask = trajectories.state[0].to(args.device), trajectories.state[1:].to(args.device), \
    #                                     trajectories.pad[1:].to(args.device)
    #         optim.zero_grad()
    #         pred_y = odeint(func, y0, t, method=args.method)
    #         loss = ((pred_y - trajectory) ** 2 * loss_mask).mean()
    #         loss.backward()
    #         optim.step()
    #
    #         """ plot """
    #         print(loss.item())
    #
    #         if cooldown():
    #             func.h = trajectories.action.to(args.device)
    #             t = torch.linspace(0, len(trajectories.state) - 2, len(trajectories.state) - 1, device=args.device)
    #             y0, trajectory = trajectories.state[0].to(args.device), trajectories.state[1:].to(args.device)
    #             pred_y = odeint(func, y0, t, method=args.method)
