from data.datasets import SARDataset, gather_data
from viz import debug_image
from functional import gaussian_like_function, multivariate_gaussian, multivariate_diag_gaussian
from data.utils import chomp, pad_collate
import gym
from env import wrappers
from atariari.benchmark.wrapper import AtariARIWrapper
import numpy as np
import pytest
from torch.utils.data import DataLoader
import torch
from pyrr import rectangle
from math import pi

@pytest.fixture
def buffer():
    env = gym.make('PongNoFrameskip-v4')
    env = wrappers.NoopResetEnv(env, noop_max=30)
    env = wrappers.MaxAndSkipEnv(env, skip=4)
    env = AtariARIWrapper(env)
    env = wrappers.AtariAriVector(env)
    env = wrappers.FireResetEnv(env)
    return gather_data(2, env)


def test_sar_dataset(buffer):
    data = SARDataset(buffer)
    state, action, reward = data[0]


def test_pad_collateSAR(buffer):
    data = SARDataset(buffer)
    data = DataLoader(data, batch_size=2, collate_fn=pad_collate)

    for mb in data:
        state, action, reward, mask = mb.state, mb.action, mb.reward, mb.mask
        source = torch.cat((state, action, reward), dim=2)
        #target = state.narrow(dim=2, start=2, length=2)
        target = torch.cat((state, action, reward), dim=2)
        source = chomp(source, 'right', dim=1, bite_size=1)
        target = chomp(target, 'left', dim=1, bite_size=1)
        mask = chomp(mask, 'left', dim=1, bite_size=1)

        assert source[0][1][0] == target[0][0][0]
        assert torch.allclose(source[0, 1, :], target[0, 0, :])
        assert torch.allclose(source[:, 1:, :], target[:, 0:target.size(1)-1, :])
        assert torch.allclose(torch.sum(target * mask), torch.sum(target))
        mask_rolled = torch.cat((torch.ones_like(mask[:, 0, :].unsqueeze(1)), mask[:, :-1, :]), dim=1)
        assert torch.allclose(torch.sum(source * mask_rolled), torch.sum(source))


def test_rect():
    rect = rectangle.create()
    debug_image(rect, block=True)


def test_gaussian():
    pos = []
    labels = [(0.1, 0.1)]
    for y, x in labels:
        pos.append(np.array([y, x]))
    pos = np.stack(pos, axis=0)
    probmap = gaussian_like_function(pos, 240, 240, sigma=0.2)
    image = (probmap * 255).astype(np.uint)
    debug_image(image, block=True)


# def gaussian_like_function(kp, height, width, sigma=0.1, eps=1e-6):
#     hm = squared_diff(kp[:, 0], height)
#     wm = squared_diff(kp[:, 1], width)
#     hm = hm[:, :, np.newaxis]
#     hm = np.repeat(hm, repeats=width, axis=2)
#     wm = wm[:, np.newaxis, :]
#     wm = np.repeat(wm, repeats=height, axis=1)
#     gm = - np.sqrt(hm + wm + eps) / (2 * sigma ** 2)
#     gm = np.exp(gm)
#     return gm


def gaussian2Dd(mu, sigma, size):

    D = mu.size(0)
    axes = [torch.linspace(0, 1.0, size) for _ in range(D)]
    grid = torch.meshgrid(axes)
    grid = torch.stack(grid)
    gridshape = grid.shape[1:]
    grid = grid.flatten(start_dim=1)
    eps = torch.finfo(mu.dtype).eps
    sigma = sigma + eps

    constant = 1 / torch.sqrt(2 * pi * torch.prod(sigma))
    #
    N = grid.size(1)
    m = grid - mu.view(-1, 1)
    m = m.view(-1, 1, D)
    inv_covar = torch.diag_embed(1 / sigma)
    exponent = - 0.5 * torch.matmul(m, inv_covar).matmul(m.permute(0, 2, 1))
    gaussian = constant * torch.exp(exponent)
    return gaussian.squeeze().reshape(1, *gridshape)

    # axes = []
    # for axis in grid:
    #     exponent = -1 * (x_axis - mu) ** 2 / (2 * sigma ** 2)
    #
    #     y = constant * torch.exp(exponent)
    #     axes.append(y)

    # h = torch.sum(torch.stack(axes), dim=0)
    # gm = -torch.sqrt(torch.cumprod())

    # return gm


def gaussian2dw(mu, sigma, size):
    D = mu.size(0)
    axes = [torch.linspace(0, 1.0, size) for _ in range(D)]
    x, y = torch.meshgrid(axes)
    #grid = torch.stack(grid)
    #gridshape = grid.shape[1:]
    #grid = grid.flatten(start_dim=1)
    eps = torch.finfo(mu.dtype).eps
    sigma = sigma + eps

    constant = 1 / torch.sqrt(2 * pi * torch.prod(sigma))

    x, y = (x - mu[0]) ** 2 / (2 * sigma[0] ** 2), (y - mu[1]) ** 2 / (2 * sigma[1] ** 2)
    exponent = torch.exp(- x - y)
    return constant * exponent


def test_multivariate_diag_gaussian():
    mu = torch.tensor([[0.5, 0.5]])
    stdev = torch.tensor([[0.1, 0.3]])

    image = multivariate_diag_gaussian(mu, stdev, (40, 20))
    image = image.numpy()
    image = (image / image.max() * 255).astype(np.uint)
    debug_image(image, block=True)


def test_multivariate_gaussian():

    mu = torch.tensor([[0.5, 0.5]])
    covar = torch.tensor([[0.1, 0.0] , [0.0, 0.1]])

    image = multivariate_gaussian(mu, covar, (40, 20))
    image = image.numpy()
    image = (image / image.max() * 255).astype(np.uint)
    debug_image(image, block=True)

    # trajectory = buffer.trajectories[0]
    #
    # assert len(trajectory) - 1 == source.shape[0]
    #
    # assert np.allclose(trajectory[0].state, source[0, 0, 0:4])
    # assert np.allclose(trajectory[0].reward, source[0, 0, 4])
    # assert np.allclose(trajectory[0].state, source[0, 1, 0:4])
    # assert np.allclose(trajectory[0].reward, source[0, 1, 4])
    #
    # state, reward, done = trajectory[1].info['alternates'][0]
    # assert np.allclose(state, target[0, 0, 0:4])
    # assert np.allclose(reward, target[0, 0, 4])
    # state, reward, done = trajectory[1].info['alternates'][1]
    # assert np.allclose(state, target[0, 1, 0:4])
    # assert np.allclose(reward, target[0, 1, 4])
    #
    # final = len(trajectory) - 2
    # assert np.allclose(trajectory[final].state, source[final, 0, 0:4])
    # assert np.allclose(trajectory[final].reward, source[final, 0, 4])
    # assert np.allclose(trajectory[final].state, source[final, 1, 0:4])
    # assert np.allclose(trajectory[final].reward, source[final, 1, 4])
    #
    # state, reward, done = trajectory[final + 1].info['alternates'][0]
    # assert np.allclose(state, target[final, 0, 0:4])
    # assert np.allclose(reward, target[final, 0, 4])
    #
    # state, reward, done = trajectory[final + 1].info['alternates'][1]
    # assert np.allclose(state, target[final, 1, 0:4])
    # assert np.allclose(reward, target[final, 1, 4])
    #
    # assert np.allclose(one_hot(0, 5), action[0, 0, :])
    # assert np.allclose(one_hot(1, 5), action[0, 1, :])
    # assert np.allclose(one_hot(5, 5), action[0, 5, :])
