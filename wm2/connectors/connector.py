import numpy as np
import torch
from torch.distributions import OneHotCategorical, Normal

from models.transition import LSTMTransitionModel, ODEDynamicsModel
from wm2.distributions import ScaledTanhTransformedGaussian
from wm2.models.models import SoftplusMLP, MLP, Policy
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


class RandomPolicy(nn.Module):
    def __init__(self, action_dims):
        super().__init__()
        self.action_dims = action_dims

    def forward(self, state):
        mu = torch.zeros((1, self.action_dims,))
        scale = torch.full((1, self.action_dims,), 0.5)
        return ScaledTanhTransformedGaussian(mu, scale)


class RandomDiscretePolicy(nn.Module):
    def __init__(self, action_dims):
        super().__init__()
        self.action_dims = action_dims

    def forward(self, state):
        return OneHotCategorical(probs=torch.ones(1, self.action_dims)/self.action_dims)


def no_explore(args, action_dist):
    return action_dist


def no_sample(action):
    return action


class ActionPipeline:
    def __init__(self, policy_prepro, env_action_prepro, explore=None, sample=None):
        self.policy_prepro = policy_prepro
        self.explore = no_explore if explore is None else explore
        self.sample = no_sample if sample is None else sample
        self.env_action_prepro = env_action_prepro

    def __call__(self, args, state, policy):
        pre_state = self.policy_prepro(state, args.device)
        action_dist = policy(pre_state)
        action_dist = self.explore(args, action_dist)
        sampled_action = self.sample(action_dist)
        action = self.env_action_prepro(sampled_action)
        return action_dist, sampled_action, action


class EnvConnector:

    @staticmethod
    def policy_prepro(state, device):
        return torch.tensor(state).unsqueeze(0).float().to(device)

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
    def explore(args, action_dist):
        action = action_dist.rsample()
        return Normal(action, args.exploration_noise)

    @staticmethod
    def sample(action_dist):
        return action_dist.rsample()

    @staticmethod
    def store_action_prepro(args, action):
        return action

    @staticmethod
    def make_action_pipeline():
        return ActionPipeline(policy_prepro=EnvConnector.policy_prepro,
                              env_action_prepro=EnvConnector.action_prepro,
                              explore=EnvConnector.explore,
                              sample=EnvConnector.sample
                              )

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
        if isinstance(env.action_space, gym.spaces.Discrete):
            return RandomDiscretePolicy(env.action_space.n)
        elif isinstance(env.action_space, gym.spaces.Box):
            return RandomPolicy(env.action_space.shape[0])

    def set_env_dims(self, args, env):
        args.state_dims = env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.Discrete):
            args.action_dims = env.action_space.n
            args.action_min = 0
            args.action_max = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
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
        return LSTMTransitionModel(args)

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



class ODEEnvConnector(EnvConnector):

    @staticmethod
    def make_transition_model(args):
        return ODEDynamicsModel(args)
