import torch
import wandb
from torch import distributions as dist
from torch.distributions import kl_divergence
from torch.utils.data import DataLoader

from data.datasets import SARNextDataset, SARDataset
from data.utils import pad_collate_2
from models.models import StochasticTransitionModel, ForcedDynamics, DreamTransitionModel
from torchdiffeq import odeint_adjoint as odeint
from utils import Cooldown
from viz import VizTransition


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