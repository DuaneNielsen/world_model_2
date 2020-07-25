import numpy as np
import torch
from wm2.distributions import ScaledTanhTransformedGaussian
import torch.distributions as dist
from wm2.models.models import SoftplusMLP, MLP, StochasticTransitionModel


class EnvConnector:
    def __init__(self, **kwargs):
        """ required arguments, state_dims, action_dims """
        self.state_dims = None
        self.action_dims = None

        self.__dict__.update(**kwargs)

        if self.state_dims is None:
            raise Exception('state_dims not set on connector, fix the observation_space on env')
        if self.action_dims is None:
            raise Exception('action_dims not set on connector, fix the action_space on env')

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
    def make_transition_model(args):
        return StochasticTransitionModel(input_dim=args.state_dims + args.action_dims,
                                  hidden_dim=args.dynamics_hidden_dim, output_dim=args.state_dims,
                                  layers=args.dynamics_layers)

    @staticmethod
    def make_pcont_model(args):
        return SoftplusMLP([args.state_dims, *args.pcont_hidden_dims, 1])

    @staticmethod
    def make_reward_model(args):
        return MLP([args.state_dims, *args.reward_hidden_dims, 1], nonlin=args.reward_nonlin)