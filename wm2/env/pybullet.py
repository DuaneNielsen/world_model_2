import torch

from env.connector import EnvConnector
import numpy as np
import gym
import torch.nn as nn
import torch.nn.functional as F


class DiffReward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, state):
        state = torch.transpose(state, 0, -1)
        done_flag = state[-1]
        target_dist = state[-2]
        speed = state[-3]
        # target_dist = target_dist.clamp(max=0.999, min=0.0)
        # reward = (1.0 - done_flag + (0.95 ** (target_dist * 1000.0 / 20.0)) * (1.0 - done_flag)).unsqueeze(0)
        # reward = (1.0 - state[-1]) / ((1.0 - state[-2]).sqrt() + torch.finfo(state).eps)
        eps = torch.finfo(speed.dtype).eps
        # reward = ((1.5 - torch.log(1.5 - target_dist)) * (1.0 - done_flag)).unsqueeze(0)
        # reward = (1.0 - done_flag)
        # reward = 20 / (1 + torch.exp((target_dist - 0.7) * 10)) * (1.0-done_flag)
        reward = F.leaky_relu(speed) * self.args.forward_slope
        # position = (1.0 - target_dist) * args.forward_slope
        # reward = reward * (1.0 - done_flag)
        reward = reward.unsqueeze(0)
        reward = torch.transpose(reward, 0, -1)
        return reward


class PybulletWalkerWrapper(gym.Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        low = np.concatenate((self.unwrapped.observation_space.low, np.array([-np.inf, -np.inf, 0])))
        high = np.concatenate((self.unwrapped.observation_space.high, np.array([+np.inf, np.inf, 1])))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.robot = env.unwrapped.robot
        self.prev_target_dist = None
        self.args = args

    def reset(self):
        state = self.env.reset()
        self.prev_target_dist = np.array([self.robot.walk_target_dist], dtype=state.dtype) / 1000.0
        forward_speed = np.array([0.0], dtype=state.dtype)
        target_dist = np.array([self.robot.walk_target_dist], dtype=state.dtype) / 1000.0
        done_flag = np.full(1, fill_value=0, dtype=state.dtype)
        return np.concatenate((state, forward_speed, target_dist, done_flag), axis=0)

    def step(self, action):
        raw_state, rew, done, info = self.env.step(action)
        target_dist = np.array([self.robot.walk_target_dist], dtype=raw_state.dtype) / 1000.0
        speed = self.prev_target_dist - target_dist
        self.prev_target_dist = target_dist
        done_flag = np.full(1, fill_value=done, dtype=raw_state.dtype)
        state = np.concatenate((raw_state, speed, target_dist, done_flag), axis=0)
        # if done:
        #     rew = 0.0
        # else:
            #rew = 1.0 + 0.95 ** (self.robot.walk_target_dist / 20.0)
        #rew = (1.5 - np.log(1.5 - target_dist)) * (1-done_flag)
        # rew = 20/(1+np.exp((target_dist - 0.7)*10)) * (1.0-done_flag)

        speed = speed if speed > 0.0 else speed * 1e-2
        #target_dist = 1.0 if target_dist > 1.0 else target_dist
        rew = speed * self.args.forward_slope
        rew = rew * (1.0 - done_flag)
        return state, rew.item(), done, info


class PyBulletEnv(EnvConnector):
    def __init__(self, **kwargs):
        super().__init__()

    def make_env(self, args):
        env = gym.make(args.env)
        self.set_env_dims(args, env)
        env.render()
        return env


class PyBulletCheetahConnector(EnvConnector):
    def __init__(self, **kwargs):
        super().__init__()

    def make_env(self, args):
        # environment
        env = gym.make(args.env)
        self.set_env_dims(args, env)
        # env = wm2.env.wrappers.ConcatPrev(env)
        # env = wm2.env.wrappers.AddDoneToState(env)
        # env = wm2.env.wrappers.RewardOneIfNotDone(env)
        env = PybulletWalkerWrapper(env, args)
        env.render()
        return env

    def make_reward_model(self, args):
        return DiffReward(args)

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
