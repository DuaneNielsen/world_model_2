from train_model import SARDataset
from train_model import gather_data, pad_collate, chomp
from utils import gaussian_like_function, debug_image
import gym
from env import wrappers
from atariari.benchmark.wrapper import AtariARIWrapper
import numpy as np
import pytest
from torch.utils.data import DataLoader
import torch
from pyrr import rectangle

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
