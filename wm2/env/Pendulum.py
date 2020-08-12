import numpy as np
import torch
from matplotlib import pyplot as plt
from connectors.connector import EnvConnector, EnvViz


class PendulumConnector(EnvConnector):
    def __init__(self, **kwargs):
        super().__init__()

    # @staticmethod
    # def action_prepro(action):
    #     return np.array([action.item()], dtype=np.float32)

    @staticmethod
    def make_viz(args):
        return PendulumViz()


class PendulumViz(EnvViz):
    def __init__(self):
        # visualization
        plt.ion()
        self.fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
        self.polar = self.fig.add_subplot(111, projection='polar')
        self.theta = np.arange(0, np.pi * 2, 0.01, dtype=np.float32)[:, np.newaxis]
        self.speeds = np.linspace(-8.0, 8.0, 7, dtype=np.float32)
        self.speedlines = []
        for speed in self.speeds:
            self.speedlines += self.polar.plot(self.theta, np.ones_like(self.theta), label=f'{speed.item()}')
        self.polar.grid(True)
        self.polar.legend()
        self.polar.set_theta_zero_location("N")
        self.polar.relim()
        self.polar.autoscale_view()
        self.fig.canvas.draw()

        # self.polar.set_rmax(2)
        # self.polar.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        # self.polar.set_rlabel_position(-22.5)  # Move radial labels away from plotted line

    def update(self, args, test_buff, policy, R, value, T, pcont):
        with torch.no_grad():
            for i, speed in enumerate(self.speeds):
                theta = np.arange(0, np.pi * 2, 0.01, dtype=np.float32)[:, np.newaxis]
                x, y, thetadot = np.cos(theta), np.sin(theta), np.ones_like(theta) * speed
                plot_states = np.concatenate((x, y, thetadot), axis=1)
                plot_states = torch.from_numpy(plot_states).to(args.device)
                plot_v = value(plot_states)
                plot_v = plot_v.detach().cpu().numpy()
                self.speedlines[i].set_data(theta, plot_v)

            self.polar.relim()
            self.polar.autoscale_view()
            self.fig.canvas.draw()