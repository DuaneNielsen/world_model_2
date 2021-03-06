import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.init
from data.utils import chomp


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


def init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform(m.weight)
    if m.bias is not None:
        torch.nn.init.kaiming_uniform(m.bias)


def init(m, init_func):
    if isinstance(m, nn.Conv1d):
        init_func(m.weight)
        if m.bias is not None:
            init_func(m.bias)
    if isinstance(m, nn.Linear):
        init_func(m.weight)
        if m.bias is not None:
            init_func(m.bias)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, nonlin=None):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU() if nonlin is None else nonlin()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU() if nonlin is None else nonlin()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU() if nonlin is None else nonlin()
        self.init_weights()

    def init_weights(self):

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    An Empirical Evaluation of Generic Convolution and Recurrent Networks for Sequence Modeling
    https://arxiv.org/pdf/1803.01271.pdf
    https://github.com/locuslab/TCN
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, nonlin=None, final_nonlin=None):
        """
        :param num_inputs: the number of time series signals to input
        :param num_channels: the number of filters in each layer
        :param kernel_size: the time-steps to combine at each layer
        :param dropout: dropout frequency
        :param nonlin: nonlinearity to apply at each layer (default ReLU)
        :param final_nonlin: nonlinearity to apply at final layer (default ReLU)
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            if i == num_levels -1 and final_nonlin is not None:
                nonlin = final_nonlin
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, nonlin=nonlin)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Causal(nn.Module):
    def __init__(self, state_dims, action_dims, reward_dims, hidden_layers, output_dims):
        super().__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.reward_dims = reward_dims
        self.hidden_layers = hidden_layers
        self.output_dims = output_dims
        input_channels = self.state_dims + self.action_dims + self.reward_dims
        hidden_layers += [output_dims]
        self.tcn = TemporalConvNet(input_channels, hidden_layers)

    def forward(self, state, action):
        inp = torch.cat((state, action), dim=2)
        inp = inp.permute(0, 2, 1)
        out = self.tcn(inp)
        return out.permute(0, 2, 1)


class RewardNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, reward_dim, scale=1.0, offset=0.0):
        super().__init__()
        hidden_layers += [reward_dim]
        self.tcn =TemporalConvNet(input_dim, hidden_layers, nonlin=nn.Tanh)
        self.scale = scale
        self.offset = offset

    def forward(self, inp):
        inp = inp.permute(0, 2, 1)
        z = self.tcn(inp)
        return z.permute(0, 2, 1) * self.scale


class DoneNet(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        hidden_layers += [1]
        self.tcn =TemporalConvNet(input_dim, hidden_layers, final_nonlin=nn.Sigmoid)

    def forward(self, inp):
        inp = inp.permute(0, 2, 1)
        z = self.tcn(inp)
        return z.permute(0, 2, 1)


class Encoder(nn.Module):
    def __init__(self, state_dims, action_dims, reward_dims, hidden_layers, output_dims, nonlin=None):
        super().__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.reward_dims = reward_dims
        self.hidden_layers = hidden_layers
        self.output_dims = output_dims
        input_channels = self.state_dims + self.action_dims + self.reward_dims
        hidden_layers += [output_dims]
        self.tcn = TemporalConvNet(input_channels, hidden_layers, nonlin=nonlin)

    def forward(self, inp):
        """
        :param inp: tensor of shape (N, T, D) : N = Batch, T=Time, D=dimensions
        :return:
        """
        inp = inp.permute(0, 2, 1)
        z = self.tcn(inp)
        return z.permute(0, 2, 1)


class Decoder(nn.Module):
    def __init__(self, state_dims, action_dims, reward_dims, hidden_state_dims, target_len, layers=2, dropout=0.2):
        super().__init__()
        input_channels = state_dims + action_dims + reward_dims
        self.layers = layers
        self.lstm = nn.LSTM(input_channels, hidden_state_dims, layers, dropout=dropout)
        self.output_block = nn.Linear(hidden_state_dims, target_len)

    def forward(self, i, h):
        o, h = self.lstm(i, h)
        s = self.output_block(o)
        return s, h


class TCMDN(nn.Module):
    def __init__(self, state_dims, action_dims, reward_dims, hidden_layers, output_dims):
        super().__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.reward_dims = reward_dims
        self.hidden_layers = hidden_layers
        self.output_dims = output_dims
        input_channels = self.state_dims + self.action_dims + self.reward_dims
        hidden_layers += [output_dims * 2]
        self.tcn = TemporalConvNet(input_channels, hidden_layers)
        self.mu_net = nn.Linear(output_dims, output_dims, bias=False)
        self.stdev_net = nn.Linear(output_dims, output_dims, bias=False)

    def forward(self, state, action):
        inp = torch.cat((state, action), dim=2)
        inp = inp.permute(0, 2, 1)
        out = self.tcn(inp)
        out = out.permute(0, 2, 1)
        mu, stdev = out.chunk(2, dim=2)
        mu = self.mu_net(mu)
        stdev = self.stdev_net(stdev)
        return mu, stdev
