import torch
from distributions import ScaledTanhTransformedGaussian
import numpy as np


class PyBulletConnector:
    def __init__(self, action_dims):
        self.action_dims = action_dims

    @staticmethod
    def policy_prepro(state, device):
        return torch.tensor(state).float().to(device)

    @staticmethod
    def buffer_prepro(state):
        return state.astype(np.float32)

    @staticmethod
    def reward_prepro(reward):
        return np.array([reward], dtype=np.float32)

    @staticmethod
    def action_prepro(action):
        if action.shape[1] == 1:
            return action.detach().cpu().squeeze().unsqueeze(0).numpy().astype(np.float32)
        else:
            return action.detach().cpu().squeeze().numpy().astype(np.float32)

    def random_policy(self, state):
        mu = torch.zeros((1, self.action_dims,))
        scale = torch.full((1, self.action_dims,), 0.5)
        return ScaledTanhTransformedGaussian(mu, scale)

    @staticmethod
    def reward_mask_f(state, reward, action):
        r = np.concatenate(reward)
        less_than_0_3 = r <= 0.3
        p = np.ones_like(r)
        num_small_reward = r.shape[0] - less_than_0_3.sum()
        if num_small_reward > 0:
            p = p / num_small_reward
            p = p * ~less_than_0_3
        else:
            p = p / r.shape[0]
        i = np.random.choice(r.shape[0], less_than_0_3.sum(), p=p)
        less_than_0_3[i] = True
        return less_than_0_3[:, np.newaxis]