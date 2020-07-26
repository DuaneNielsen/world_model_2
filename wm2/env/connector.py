import numpy as np
import torch
from wm2.distributions import ScaledTanhTransformedGaussian
import torch.distributions as dist
from wm2.models.models import SoftplusMLP, MLP, StochasticTransitionModel
import gym
from torch import nn

class EnvViz:

    def update(self, args, s, policy, R, value, T, pcont):
        pass

class RandomPolicy:
    def __init__(self, action_dims):
        super().__init__()
        self.action_dims = action_dims

    def __call__(self, state):
        mu = torch.zeros((1, self.action_dims,))
        scale = torch.full((1, self.action_dims,), 0.5)
        return ScaledTanhTransformedGaussian(mu, scale)


class EnvConnector:

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

    @staticmethod
    def make_random_policy(env):
        return RandomPolicy(env.action_space.shape[0])

    def set_env_dims(self, args, env):
        args.state_dims = env.observation_space.shape[0]
        args.action_dims = env.action_space.shape[0]
        args.action_min = -1.0
        args.action_max = 1.0

    def make_env(self, args):
        # environment
        env = gym.make(args.env)
        self.set_env_dims(args, env)
        # env = wm2.env.wrappers.ConcatPrev(env)
        # env = wm2.env.wrappers.AddDoneToState(env)
        # env = wm2.env.wrappers.RewardOneIfNotDone(env)
        # env = wm2.env.pybullet.PybulletWalkerWrapper(env, args)
        env.render()

        # env = gym.wrappers.TransformReward(env, alwaysone)

        # env.connector = LunarLanderConnector
        return env

    @staticmethod
    def make_transition_model(args):
        return StochasticTransitionModel(input_dim=args.state_dims + args.action_dims,
                                  hidden_dim=args.dynamics_hidden_dim, output_dim=args.state_dims,
                                  layers=args.dynamics_layers, dropout=args.dynamics_dropout)

    @staticmethod
    def make_pcont_model(args):
        return SoftplusMLP([args.state_dims, *args.pcont_hidden_dims, 1])

    @staticmethod
    def make_reward_model(args):
        return MLP([args.state_dims, *args.reward_hidden_dims, 1], nonlin=args.reward_nonlin)

    @staticmethod
    def make_viz(args):
        return EnvViz()