from pathlib import Path

import torch
from torch import nn as nn
from torch.distributions import Categorical, Normal, MixtureSameFamily, OneHotCategorical
from torch.nn import functional as F

from distributions import ScaledTanhTransformedGaussian
import wm2.utils


class MLP(nn.Module):
    def __init__(self, layers, nonlin=None, dropout=0.2):
        super().__init__()
        in_dims = layers[0]
        net = []
        # the use of eval here is a security bug
        nonlin = nn.ELU if nonlin is None else eval(nonlin)
        for hidden in layers[1:-1]:
            net += [nn.Linear(in_dims, hidden)]
            net += [nn.Dropout(dropout)]
            net += [nonlin()]
            in_dims = hidden
        last = nn.Linear(in_dims, layers[-1], bias=False)
        #nn.init.zeros_(last.weight)
        net += [last]

        self.mlp = nn.Sequential(*net)



    def forward(self, inp):
        return self.mlp(inp)


class Mixture(nn.Module):
    def __init__(self, state_dims, hidden_dims, n_gaussians=12, nonlin=None):
        super().__init__()
        self.hidden_net = MLP([state_dims, *hidden_dims], nonlin)
        n_hidden = hidden_dims[-1]
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

    def forward(self, inp):
        last_dim = len(inp.shape) - 1
        hidden = torch.tanh(self.hidden_net(inp))
        pi = torch.softmax(self.z_pi(hidden), dim=last_dim)
        mu = self.z_mu(hidden)
        sigma = torch.exp(self.z_sigma(hidden))
        mix = Categorical(probs=pi)
        comp = Normal(mu, sigma)
        gmm = MixtureSameFamily(mix, comp)
        return gmm


class SoftplusMLP(nn.Module):
    def __init__(self, layers, nonlin=None):
        super().__init__()
        self.mlp = MLP(layers, nonlin)

    def forward(self, input):
        return F.softplus(self.mlp(input))


class Policy(nn.Module):
    def __init__(self, layers, min=-1.0, max=1.0, nonlin=None):
        super().__init__()
        self.mu = MLP(layers, nonlin, dropout=0.0)
        self.scale = nn.Linear(layers[0], 1, bias=False)
        self.min = min
        self.max = max

    def forward(self, state):
        mu = self.mu(state)
        scale = torch.sigmoid(self.scale(state)) + 0.1
        return ScaledTanhTransformedGaussian(mu, scale, min=self.min, max=self.max)


class StraightThroughOneHot(torch.autograd.Function):
    """
    Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    https://arxiv.org/pdf/1308.3432.pdf
    """
    @staticmethod
    def forward(ctx, x):
        out = torch.zeros_like(x)
        argm = torch.argmax(x, dim=1)
        out[torch.arange(x.shape[0]), argm] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad


class StraightThroughOneHotSample(torch.autograd.Function):
    """
    Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    https://arxiv.org/pdf/1308.3432.pdf

    with sampling
    """
    @staticmethod
    def forward(ctx, x):
        head, tail = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, tail).clone()
        index = torch.multinomial(x, 1).squeeze()
        #index = torch.argmax(x, dim=1)
        x[:, :] = 0.0
        x[torch.arange(x.shape[0]), index] = 1.0
        x = x.reshape(*head, tail)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class DiscretePolicy(nn.Module):
    def __init__(self, layers, nonlin=None):
        super().__init__()
        self.mu = MLP(layers, nonlin, dropout=0.0)
        self.stoh = StraightThroughOneHotSample()

    def forward(self, state):
        p = F.softmax(self.mu(state), dim=-1)
        return self.stoh.apply(p)


class TransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, dropout=dropout)
        self.outnet = nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=False),
                                    nn.Dropout(dropout))

    def dropout_off(self):
        self._force_dropout(0)

    def dropout_on(self):
        self._force_dropout(self.dropout)

    def _force_dropout(self, dropout):
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

            elif isinstance(module, nn.LSTM):
                module.dropout = dropout

            elif isinstance(module, nn.GRU):
                module.dropout = dropout

    def forward(self, inp, hx=None):
        output, hx = self.lstm(inp, hx)
        output = self.outnet(output)
        return output, hx


class StochasticTransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim * 2, layers, dropout=dropout)
        self.mu = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout))
        self.sig = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout), nn.Sigmoid())

    def forward(self, inp, hx=None):
        hidden, hx = self.lstm(inp, hx)
        mu, sig = hidden.chunk(2, dim=2)
        mu, sig = self.mu(mu), self.sig(sig) + 0.1
        output_dist = Normal(mu, sig)
        return output_dist, hx

    def dropout_off(self):
        self._force_dropout(0)

    def dropout_on(self):
        self._force_dropout(self.dropout)

    def _force_dropout(self, dropout):
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

            elif isinstance(module, nn.LSTM):
                module.dropout = dropout

            elif isinstance(module, nn.GRU):
                module.dropout = dropout


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Dropout(dropout), nn.ELU())
        self.cell = nn.GRUCell(hidden_dim, hidden_dim * 2)
        self.mu = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout), nn.ELU())
        self.sig = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Dropout(dropout), nn.Softplus())

    def forward(self, input, hx=None):
        output = []

        for step in input:
            step = self.enc(step)
            hx = self.cell(step, hx)
            output.append(hx)
        hidden = torch.stack(output)
        mu, sig = hidden.chunk(2, dim=2)
        mu, sig = self.mu(mu), self.sig(sig) + 0.1
        return Normal(mu, sig), hidden


class Dynamics(nn.Module):
    def __init__(self, state_size=6, action_size=2, nhidden=512):
        super().__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(state_size + action_size, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, state_size, bias=False)
        torch.nn.init.zeros_(self.fc3.weight.data)

    def forward(self, controlled):
        hidden = self.fc1(controlled)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.elu(hidden)
        dstate = self.fc3(hidden)
        return dstate


class ForcedDynamics(nn.Module):
    """
    in learn_dynamics mode, set h to be the trajectory of actions taken, shape L, N, A
    L = length of trajectory, N is batch size, A is action space dimensions
    """

    def __init__(self, state_size=6, action_size=2, nhidden=512, sample_func=None):
        super().__init__()
        self.dynamics = Dynamics(state_size=state_size, action_size=action_size, nhidden=nhidden)
        self.nfe = 0
        self.policy = None
        self.mode = 'learn_dynamics'
        self.h = None
        self.sample_func = sample_func

    def forward(self, t, state):

        self.nfe += 1

        if self.mode == 'learn_dynamics':
            index = torch.floor(t).long()
            index = index.clamp(0, self.h.shape[0] - 1)
            actions = self.h[index]
        else:
            if self.sample_func is not None:
                actions = self.sample_func(self.policy(state))
            else:
                actions = self.policy(state).rsample()

        controlled = torch.cat([state, actions], dim=1)
        dstate = self.dynamics(controlled)
        return dstate


class DreamModel(nn.Module):
    def __init__(self, name, model, *args, **kwargs):
        super().__init__()
        self.saver = wm2.utils.SaveLoad(name)
        self.model = model

    def forward(self, *args, **kwargs):
        self.model.forward(*args, **kwargs)

    def learn(self, args, buffer, optim):
        pass

    def save(self, suffix, **kwargs):
        """ saves model state dict as a dict['model'] to a file with <model_name>_suffix.pt"""
        self.saver.save(self, suffix, **kwargs)

    @staticmethod
    def load(wandb_run_dir, label, suffix):
        return wm2.utils.SaveLoad.load(wandb_run_dir, label, suffix)

    def checkpoint(self, optimizer):
        self.saver.checkpoint(self.model, optimizer)

    @staticmethod
    def restore_checkpoint(label, wandb_run_dir):
        f = str(Path(wandb_run_dir) / Path(f'{label}_checkpoint.pt'))
        return torch.load(f)


class DreamTransitionModel(DreamModel):
    def imagine(self, args, trajectory, policy, action_pipeline):
        """
        predicts args.horizon steps ahead using T and initial conditions sampled from exp buffer
        :param args, configuration namespace, includes args.horizon, the expented return length
        :param trajectory: trajectory namespace from pad_collate_2
        :return: H, N, (state): imagined trajectory of states
        """
        pass