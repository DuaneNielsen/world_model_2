import numpy as np
import torch
from pathlib import Path
import wandb
from wm2.distributions import ScaledTanhTransformedGaussian
import torch.distributions as dist
from wm2.models.models import SoftplusMLP, MLP, StochasticTransitionModel, Policy, ForcedDynamics
import gym
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from wm2.data.datasets import SARDataset, SARNextDataset
from wm2.data.utils import pad_collate_2
from wm2.viz import VizTransition
from wm2.utils import Cooldown
from torchdiffeq import odeint_adjoint as odeint
from torch.distributions.kl import kl_divergence
import wm2.utils

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


class DreamModel(nn.Module):
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self.model = None
        self.saver = wm2.utils.SaveLoad(name)

    def learn(self, args, buffer, optim):
        pass

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def save(self, suffix, **kwargs):
        """ saves model state dict as a dict['model'] to a file with <model_name>_suffix.pt"""
        self.saver.save(self, suffix, **kwargs)

    @staticmethod
    def load(self, wandb_run_dir, label, suffix):
        return wm2.utils.SaveLoad.load(wandb_run_dir, label, suffix)

    def checkpoint(self, optimizer):
        self.saver.checkpoint(self.model, optimizer)

    @staticmethod
    def restore_checkpoint(label, wandb_run_dir):
        f = str(Path(wandb_run_dir) / Path(f'{label}_checkpoint.pt'))
        return torch.load(f)


class DreamTransitionModel(DreamModel):
    def imagine(self, args, trajectory, policy):
        pass


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


def log_prob_loss(trajectories, predicted_state, args):
    prior = dist.Normal(predicted_state.loc[0:-1], predicted_state.scale[0:-1])
    posterior = dist.Normal(predicted_state.loc[1:], predicted_state.scale[1:])
    div = kl_divergence(prior, posterior).mean()
    log_p = predicted_state.log_prob(trajectories.next_state.to(args.device)).mean()
    return div * args.dynamics_reg - log_p


class LSTMTransitionModel(DreamTransitionModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = StochasticTransitionModel(input_dim=args.state_dims + args.action_dims,
                                  hidden_dim=args.dynamics_hidden_dim, output_dim=args.state_dims,
                                  layers=args.dynamics_layers, dropout=args.dynamics_dropout)
        self.viz = VizTransition(state_dims=args.state_dims, action_dims=args.action_dims, title=args.name)
        self.viz_cooldown = Cooldown(secs=10)

    def learn(self, args, buffer, optim):
        """ train the dynamics model """
        train = SARNextDataset(buffer, mask_f=None)
        train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)

        if hasattr(self.model, 'dropout'):
            self.model.dropout_on()

        # train transition model
        for trajectories in train:
            input = torch.cat((trajectories.state, trajectories.action), dim=2).to(args.device)
            optim.zero_grad()
            predicted_state, h = self.model(input)
            loss = log_prob_loss(trajectories, predicted_state, args)
            loss.backward()
            # clip_grad_norm_(parameters=T.parameters(), max_norm=100.0)
            optim.step()
            # scr.update_slot('transition_train', f'Transition training loss {loss.item()}')
            wandb.log({'transition_train': loss.item()})

            if self.viz_cooldown():
                self.viz.update(trajectories, predicted_state.loc[1:])

        if hasattr(self.model, 'dropout'):
            self.model.dropout_off()



        # test = SARNextDataset(sample_test_buff, mask_f=None)
        # test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
        # for trajectories in test:
        #     input = torch.cat((trajectories.state, trajectories.action), dim=2).to(args.device)
        #     predicted_state, h = T(input)
        #     loss = t_criterion(trajectories, predicted_state)
        #     scr.update_slot('transition_test', f'Transition test loss  {loss.item()}')
        #     wandb.log({'transition_test': loss.item()})

    def imagine(self, args, trajectory, policy):
        """
        predicts args.horizon steps ahead using T and initial conditions sampled from exp buffer
        :param N: Batch size
        :param L: Length of trajectories
        :param trajectory: trajectory namespace from pad_collate_2
        :return: prob of continuing, reward and value of dimension H, L, N, 1
        """
        # anchor on the sampled trajectory
        L, N = trajectory.state.shape[0:2]
        state = trajectory.state.reshape(1, N*L, -1)
        action = trajectory.action.reshape(1, N*L, -1)
        imagine = [state]

        for tau in range(args.horizon):
            step = torch.cat([state, action], dim=2)
            state, h = self.model(step)
            if isinstance(state, torch.distributions.Distribution):
                state = state.rsample()
            action = policy(state).rsample()
            imagine += [state]
        return torch.cat(imagine, dim=0).reshape(args.horizon + 1, L, N, -1)


class ODEDynamicsModel(DreamTransitionModel):
    def __init__(self, args):
        super().__init__(args)
        self.model = ForcedDynamics(state_size=args.state_dims, action_size=args.action_dims, nhidden=512)
        self.state_labels = None
        self.action_labels = None
        self.viz = VizTransition(state_dims=args.state_dims, action_dims=args.action_dims, title=args.name)
        self.viz_cooldown = Cooldown(secs=10)

    def learn(self, args, buffer, optim):

        train = SARDataset(buffer)
        train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
        self.model.mode = 'learn_dynamics'

        for trajectories in train:
            self.model.h = trajectories.action.to(args.device)
            t = torch.linspace(0, len(trajectories.state) - 2, len(trajectories.state) - 1, device=args.device)
            y0, trajectory = trajectories.state[0].to(args.device), trajectories.state[1:].to(args.device)
            loss_mask = trajectories.pad[1:].to(args.device)
            optim.zero_grad()
            pred_y = odeint(self.model, y0, t, method=args.dynamics_ode_method)
            loss = ((pred_y - trajectory) ** 2 * loss_mask).mean()
            loss.backward()
            optim.step()

            if self.viz_cooldown():
                self.viz.update(trajectories, pred_y, t)

    def imagine(self, args, trajectory, policy):
        """
        predicts args.horizon steps ahead using T and initial conditions sampled from exp buffer
        :param N: Batch size
        :param L: Length of trajectories
        :param trajectory: trajectory namespace from pad_collate_2
        :return: H, N, (state): of states
        """
        L, N = trajectory.state.shape[0:2]
        s0 = trajectory.state.reshape(N * L, -1)
        t = torch.linspace(0, args.horizon - 1, args.horizon)
        self.model.mode = 'use_policy'
        self.model.policy = policy
        prediction = odeint(self.model, s0, t, method=args.dynamics_ode_method)
        prediction = prediction.reshape(args.horizon, L, N, -1)
        return prediction


class ODEEnvConnector(EnvConnector):

    @staticmethod
    def make_transition_model(args):
        return ODEDynamicsModel(args)
