import numpy as np
import torch
from wm2.distributions import ScaledTanhTransformedGaussian
import torch.distributions as dist
from wm2.models.models import SoftplusMLP, MLP, StochasticTransitionModel, Policy
import gym
from torch import nn
from torch.optim import Adam


class EnvViz:

    def update(self, args, s, policy, R, value, T, pcont):
        pass


class PcontFixed(nn.Module):
    """ dummy function that always returns 1.0 for pcont """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = len(x.shape) - 1
        return torch.ones((*x.shape[0:shape], 1), device=x.device)


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
    def make_policy(args):
        # policy model
        policy = Policy(layers=[args.state_dims, *args.policy_hidden_dims, args.action_dims], min=args.action_min,
                        max=args.action_max, nonlin=args.policy_nonlin).to(args.device)
        policy_optim = Adam(policy.parameters(), lr=args.policy_lr)
        return policy, policy_optim

    @staticmethod
    def make_value(args):
        value = MLP([args.state_dims, *args.value_hidden_dims, 1], nonlin=args.value_nonlin).to(args.device)
        value_optim = Adam(value.parameters(), lr=args.value_lr)
        return value, value_optim

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
        # env.render()

        return env

    @staticmethod
    def make_transition_model(args):
        return StochasticTransitionModel(input_dim=args.state_dims + args.action_dims,
                                  hidden_dim=args.dynamics_hidden_dim, output_dim=args.state_dims,
                                  layers=args.dynamics_layers, dropout=args.dynamics_dropout)

    @staticmethod
    def make_pcont_model(args):
        if args.pcont_fixed_length:
            return PcontFixed(), None

        pcont = SoftplusMLP([args.state_dims, *args.pcont_hidden_dims, 1]).to(args.device)
        pcont_optim = Adam(pcont.parameters(), lr=args.pcont_lr)
        return pcont, pcont_optim

    @staticmethod
    def make_reward_model(args):
        return MLP([args.state_dims, *args.reward_hidden_dims, 1], nonlin=args.reward_nonlin)

    @staticmethod
    def make_viz(args):
        return EnvViz()