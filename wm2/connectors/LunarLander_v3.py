import gym
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from env.LunarLander_v3 import dist_k, damp_k, stablilty_k, stablity_damp, impact_k, land_k
from connectors.connector import EnvConnector, ODEEnvConnector
from env.gym_viz import VizWrapper


class PhysicsStateOnly(gym.ObservationWrapper):
    def observation(self, observation):
        return observation[0:6]


class SimpleReward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state):
        state = torch.transpose(state, 0, -1)
        x_pos, y_pos, x_velocity, y_velocity, angle, angle_velocity = state[0:6]
        eps = torch.finfo().eps
        center = (0.0, 0.0)
        dist = torch.sqrt((x_pos - center[0]) ** 2 + (y_pos - center[1]) ** 2 + eps)
        velocity = torch.sqrt(x_velocity ** 2 + y_velocity ** 2 + eps)
        reward = - dist_k * dist - damp_k * velocity - stablilty_k * torch.abs(angle) - stablity_damp * torch.abs(angle_velocity)
        reward = reward.unsqueeze(0)
        reward = torch.transpose(reward, 0, -1)
        return reward


class DiffReward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state):
        state = torch.transpose(state, 0, -1)
        x_pos, y_pos, x_velocity, y_velocity, angle, angle_velocity = state[0:6]
        eps = torch.finfo().eps
        center = (0.0, 0.0)
        stable = torch.abs(angle)
        dist = torch.sqrt((x_pos - center[0]) ** 2 + (y_pos - center[1]) ** 2 + eps)

        stable_r = - stable * stablilty_k
        dist_r = - dist_k * dist
        impact_r = - impact_k * torch.abs(y_velocity) * torch.exp(-y_pos * 2.0)
        impact_r = impact_r.clamp(min=-10.0)
        lt_x = x_pos.lt(0.1)
        gt_x = x_pos.gt(-0.1)
        lt_y = y_pos.lt(0.3)
        gt_y = y_pos.gt(-0.01)
        inside = lt_x & gt_x & lt_y & gt_y
        land_r = land_k * inside * F.relu(y_velocity + 0.25) * y_velocity.lt(0.0)

        reward = stable_r + dist_r + impact_r + land_r
        reward = reward.unsqueeze(0)
        reward = torch.transpose(reward, 0, -1)
        return reward


class LunarLanderConnector(EnvConnector):
    def __init__(self, **kwargs):
        super().__init__()

    def make_env(self, args):
        # environment
        env = gym.make(args.env)
        env = VizWrapper(env, title=args.name)
        env = PhysicsStateOnly(env)
        env.observation_space = gym.spaces.Box(-np.inf, +np.inf, (6,))
        self.set_env_dims(args, env)
        return env

    @staticmethod
    def make_reward_model(args):
        return SimpleReward()

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


class LunarLanderODEConnector(ODEEnvConnector):
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def make_reward_model(args):
        return SimpleReward()

    def make_env(self, args):
        # environment
        env = gym.make(args.env)
        env = VizWrapper(env, title=args.name)
        env = PhysicsStateOnly(env)
        env.observation_space = gym.spaces.Box(-np.inf, +np.inf, (6,))
        self.set_env_dims(args, env)
        return env


class LunarLanderDiscreteLSTMConnector(EnvConnector):
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def make_reward_model(args):
        return SimpleReward()

    def make_env(self, args):
        # environment
        env = gym.make(args.env)
        env = VizWrapper(env, title=args.name)
        env = PhysicsStateOnly(env)
        env.observation_space = gym.spaces.Box(-np.inf, +np.inf, (6,))
        self.set_env_dims(args, env)
        return env