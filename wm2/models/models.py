import torch
from torch import nn as nn
from torch.distributions import Categorical, Normal, MixtureSameFamily
from torch.nn import functional as F

from distributions import ScaledTanhTransformedGaussian


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
        net += [nn.Linear(in_dims, layers[-1], bias=False)]
        net += [nn.Dropout(dropout)]

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