import numpy as np
import gym
from connectors.connector import EnvConnector, EnvViz
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal


class LineViz(EnvViz):
    def __init__(self):
        super().__init__()
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
        self.ax1 = self.fig.add_subplot(231)
        self.ax2 = self.fig.add_subplot(232)
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(234)
        self.ax5 = self.fig.add_subplot(235)
        s = torch.linspace(-2.5, 2.5, 20).view(-1, 1)
        z = torch.zeros(20, 1)
        self.l_actions, = self.ax1.plot(s, z, 'b-', label='policy(state)')
        self.l_rewards, = self.ax2.plot(s, z, 'b-', label='reward(state)')
        self.l_next_state_0_2, = self.ax3.plot(s, z, 'b-', label='T(state,0.2)')
        self.l_next_state_minus_0_2, = self.ax3.plot(s, z, 'r-', label='T(state,-0.2)')
        self.l_value, = self.ax4.plot(s, z, 'b-', label='value(state)')
        self.prob_cont, = self.ax5.plot(s, z, 'b-', label='pcont(state)')
        self.ax1.legend(), self.ax2.legend(), self.ax3.legend(), self.ax4.legend(), self.ax5.legend()

    def update(self, args, test_buff, policy, R, value, T, pcont):

        s = torch.linspace(-2.5, 2.5, 20).view(-1, 1).to(args.device)

        a = policy.mu(s)
        r = R(s)
        v = value(s)

        s_0_2 = torch.cat((s.view(1, -1, 1), torch.full((1, 20, 1), 0.2, device=args.device)), dim=2)
        s_minus_0_2 = torch.cat((s.view(1, -1, 1), torch.full((1, 20, 1), -0.2, device=args.device)), dim=2)
        next_state_0_2_dist, hidden = T(s_0_2)
        next_state_minus_0_2_dist, hidden = T(s_minus_0_2)
        p = pcont(s)

        self.l_actions.set_ydata(a.detach().cpu().numpy())
        self.l_rewards.set_ydata(r.detach().cpu().numpy())
        self.l_next_state_0_2.set_ydata(next_state_0_2_dist.loc.detach().cpu().numpy())
        self.l_next_state_minus_0_2.set_ydata(next_state_minus_0_2_dist.loc.detach().cpu().numpy())
        self.l_value.set_ydata(v.detach().cpu().numpy())
        self.prob_cont.set_ydata(p.detach().cpu().numpy())

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax4.relim()
        self.ax4.autoscale_view()
        self.ax5.relim()
        self.ax5.autoscale_view()
        self.fig.canvas.draw()


class SimpleTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.y = nn.Linear(2, 2, bias=True)

    def forward(self, x, hx=None):
        y = self.y(x)
        mu, sig = y.chunk(2, dim=2)
        output_dist = Normal(mu, sig + 0.1)
        return output_dist, None


class LinEnvConnector(EnvConnector):
    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def make_transition_model(args):
        return SimpleTransition()

    @staticmethod
    def make_reward_model(args):
        return nn.Linear(1, 1, bias=True)

    @staticmethod
    def make_viz(args):
        return LineViz()


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

    @staticmethod
    def reward(pos):
        return (0.25 * pos - 0.5).item()

    def step(self, action):
        self.step_count += 1
        self.pos += action
        pos = self.pos.copy()

        if pos >= 2.0:
            return pos, self.reward(pos), True, {}
        elif self.step_count > 30:
            return pos, self.reward(pos), True, {}
        elif pos <= -2.0:
            return pos, self.reward(pos), True, {}
        else:
            return pos, self.reward(pos), False, {}

    def render(self, mode='human'):
        pass

