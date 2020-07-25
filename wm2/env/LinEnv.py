import numpy as np
import gym
from wm2.env.connector import EnvConnector
import torch.nn as nn


class LinEnvConnector(EnvConnector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def make_transition_model(args):
        return nn.Linear(2, 1, bias=True)

    @staticmethod
    def make_reward_model(args):
        return nn.Linear(1, 1, bias=True)
    
    @staticmethod
    def make_pcont_model(args):
        return nn.Linear(1, 1, bias=True)


class LinEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.pos = np.array([0.0], dtype=np.float32)
        self.step_count = 0
        self.observation_space = gym.spaces.Box(-2.0, +2.0, (1,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,))

    def reset(self):
        self.pos = np.array([0.0], dtype=np.float32)
        self.step_count = 0
        return self.pos.copy()

    def step(self, action):
        self.step_count += 1
        self.pos += action
        pos = self.pos.copy()

        if pos >= 2.0:
            return pos, 1.0, True, {}
        elif self.step_count > 30:
            return pos, -1.0, True, {}
        elif pos <= -2.0:
            return pos, -1.0, True, {}
        else:
            return pos, -1.0, False, {}

    def render(self, mode='human'):
        pass

