import torch
import torch.distributions as dist
from distributions import ScaledTanhTransformedGaussian
import numpy as np
import gym
import math


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


    def uniform_random_policy(self, state):
        mu = torch.ones((state.size(0), self.action_dims,))
        return dist.Uniform(-mu, mu)

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


class PybulletWalkerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.concatenate((self.unwrapped.observation_space.low, np.array([-np.inf, 0])))
        high = np.concatenate((self.unwrapped.observation_space.high, np.array([+np.inf, 1])))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.robot = env.unwrapped.robot

    def reset(self):
        state = self.env.reset()
        target_dist = np.array([self.robot.walk_target_dist], dtype=state.dtype) / 1000.0
        done_flag = np.full(1, fill_value=0, dtype=state.dtype)
        return np.concatenate((state, target_dist, done_flag), axis=0)

    def step(self, action):
        raw_state, rew, done, info = self.env.step(action)
        target_dist = np.array([self.robot.walk_target_dist], dtype=raw_state.dtype) / 1000.0
        done_flag = np.full(1, fill_value=done, dtype=raw_state.dtype)
        state = np.concatenate((raw_state, target_dist, done_flag), axis=0)
        # if done:
        #     rew = 0.0
        # else:
            #rew = 1.0 + 0.95 ** (self.robot.walk_target_dist / 20.0)
        #rew = (1.5 - np.log(1.5 - target_dist)) * (1-done_flag)
        # rew = 20/(1+np.exp((target_dist - 0.7)*10)) * (1.0-done_flag)
        rew = 10 * (1.0 - target_dist) + 0.5
        rew = rew * (1.0 - done_flag)
        return state, rew.item(), done, info