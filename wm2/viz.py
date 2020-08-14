import curses
import random
from collections import OrderedDict, deque
from statistics import mean
import math

from matplotlib import pyplot as plt
from time import sleep

from math import floor

import cv2
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint

from wm2.functional import multivariate_diag_gaussian, multivariate_gaussian
from wm2.env.runner import EnvObserver


def to_opencv_image(im):
    '''Convert to OpenCV image shape h,w,c'''
    shape = im.shape
    if len(shape) == 3 and shape[0] < shape[-1]:
        return im.transpose(1, 2, 0)
    else:
        return im


def debug_image(im, block=True):
    '''
    Renders an image for debugging; pauses process until key press
    Handles tensor/numpy and conventions among libraries
    '''
    if torch.is_tensor(im):  # if PyTorch tensor, get numpy
        im = im.cpu().numpy()
    im = to_opencv_image(im)
    im = im.astype(np.uint8)  # typecast guard
    if im.shape[0] == 3:  # RGB image
        # accommodate from RGB (numpy) to BGR (cv2)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow('debug image', im)
    if block:
        cv2.waitKey(0)
    else:
        cv2.waitKey(10)


def strip_index(center, thickness, length):
    center = floor(center * length)
    thickness = floor(thickness * length // 2)
    lower, upper = center - thickness, center + thickness
    return lower, upper


def put_strip(image_size, center, thickness, dim, mu, stdev=None, covar=None):
    image = np.zeros(image_size)
    # stdev = torch.tensor([[0.05]], device=device)
    if stdev is not None:
        strip = multivariate_diag_gaussian(mu.view(1, -1), stdev.view(1, -1), image_size)
    elif covar is not None:
        strip = multivariate_gaussian(mu.view(1, -1), covar, image_size)
    else:
        raise Exception('required either a stdev or covariance matrix')

    strip = strip.cpu().numpy()
    lower, upper = strip_index(center, thickness, image_size[dim])
    if dim == 1:
        image[:, lower:upper] = strip.T
    elif dim == 0:
        image[lower:upper, :] = strip
    image = ((image / np.max(image)) * 255).astype(np.uint)
    return image


def put_gaussian(image_size, mu, stdev=None, covar=None):
    # stdev = torch.tensor([[0.05, 0.05]], device=device)
    if stdev is not None:
        point = multivariate_diag_gaussian(mu.view(1, -1), stdev.view(1, -1), image_size)
    elif covar is not None:
        point = multivariate_gaussian(mu.view(1, -1), covar, image_size)
    else:
        raise Exception('required either a stdev or covariance matrix')
    image = point.cpu().numpy()
    image = ((image / np.max(image)) * 255).astype(np.uint)
    return image


def display_predictions(trajectory_mu, trajectory_stdev=None, trajectory_covar=None, label=None, max_length=12 * 5,
                        fps=12, scale=4, max_lookahead=3):

        length = min(trajectory_mu.size(1), max_length)
        image_size = (240 * scale, 160 * scale)

        for step in range(length):
            try:
                mu = trajectory_mu[:, step]
                covar = trajectory_covar[:, step]
                panel = []
                for t in range(max_lookahead):
                    if label == 'all':
                        player = put_strip(image_size, 0.9, 0.05, dim=1, mu=mu[0], covar=covar[0])
                        enemy = put_strip(image_size, 0.3, 0.05, dim=1, mu=mu[1], covar=covar[1])
                        ball = put_gaussian(image_size, mu[2:4], covar[2:4])
                        image = np.stack((player, enemy, ball.squeeze()))

                    elif label == 'player':
                        image = put_strip(image_size, 0.9, 0.05, dim=1, mu=mu[t], covar=covar[t])

                    elif label == 'enemy':
                        image = put_strip(image_size, 0.9, 0.05, dim=1, mu=mu[t], covar=covar[t])

                    elif label == 'ball':
                        image = put_gaussian(image_size, mu[t], covar=covar[t]).squeeze()
                    elif label == 'reward':
                        return
                    else:
                        raise Exception(f'label {label} not found')

                    panel.append(image)

                image = np.concatenate(panel, axis=1)
                debug_image(image, block=False)
                sleep(1 / fps)

            except RuntimeError:
                print('Runtime Error')


class DummyCurses:
    """  Drop this in when you want to disable curses """
    def __init__(self):
        pass

    def clear(self):
        pass

    def refresh(self):
        pass

    def update_slot(self, label, string):
        pass

    def update_progressbar(self, tics):
        pass

    def update_table(self, table, h=0, title=None):
        pass


class Curses:
    def __init__(self):
        self.stdscr = curses.initscr()

        # Clear and refresh the screen for a blank canvas
        self.stdscr.clear()
        self.stdscr.refresh()

        # Start colors in curses
        curses.start_color()
        curses.use_default_colors()

        curses.init_pair(1, 3, -1)  # slot text color
        curses.init_pair(2, 139, -1)  # status bar color

        self.height, self.width = self.stdscr.getmaxyx()

        self.table_start = {}
        self.table_insert_line = 0

        self.bar = OrderedDict()

    def _resize(self):
        resize = curses.is_term_resized(self.height, self.width)

        # Action in loop if resize is True:
        if resize is True:
            self.height, self.width = self.stdscr.getmaxyx()
            self.stdscr.clear()
            curses.resizeterm(self.height, self.width)

    def clear(self):
        self.stdscr.clear()
        self.height, self.width = self.stdscr.getmaxyx()

    def refresh(self):
        self._resize()
        self.stdscr.refresh()

    def update_slot(self, label, string):
        self.bar[label] = string
        slot = list(self.bar).index(label)
        try:
            self.stdscr.attron(curses.color_pair(1))
            self.stdscr.addstr(self.height - slot - 2, 0, self.bar[label])
            self.stdscr.addstr(self.height - slot - 2, len(self.bar[label]),
                               " " * (self.width - len(self.bar[label]) - 1))
            self.stdscr.attroff(curses.color_pair(1))
            self.refresh()
        except curses.error:
            pass

    def update_progressbar(self, tics):
        try:
            bar = '#' * tics
            self.stdscr.attron(curses.color_pair(2))
            self.stdscr.addstr(self.height - 1, 0, bar)
            self.stdscr.addstr(self.height - 1, len(bar), " " * (self.width - len(bar) - 1))
            self.stdscr.attroff(curses.color_pair(2))
            self.refresh()
        except curses.error:
            pass

    def _write_row(self, str, h=0, w=0, color_pair=0):
        try:
            self.stdscr.attron(curses.color_pair(color_pair))
            self.stdscr.addstr(h, 0, str)
            self.stdscr.addstr(h, len(str), " " * (self.width - len(str) - 1))
            self.stdscr.attroff(curses.color_pair(color_pair))
        except curses.error:
            pass

    def update_table(self, title, table):
        if len(table.shape) == 1:
            table = np.expand_dims(table, axis=0)

        rows = table.shape[0]
        if title not in self.table_start:
            self.table_start[title] = self.table_insert_line
            self.table_insert_line += rows + 1

        h = self.table_start[title]
        assert len(table.shape) == 2
        h = self.table_start[title]
        if title is not None:
            self._write_row(title, h=h)
            h = h + 1
        for i in range(rows):
            table_str = np.array2string(table[i], max_line_width=self.width)
            self._write_row(table_str, i + h)
        self.refresh()


class HistogramPanel:
    def __init__(self, fig, panels, fig_index, label):
        self.fig = fig
        self.label = label
        self.hist = fig.add_subplot(*panels, fig_index, )
        self.hist.set_title(self.label)
        self.hist.hist(np.zeros((1,)), label=label)
        self.hist.legend(label)
        self.hist.relim()
        self.hist.autoscale_view()
        self.hist.legend()

    def update(self, data, bins=100, yscale='linear'):
        """
        update the histogram
        :param data: numpy array of dim 1
        :return:
        """
        self.hist.clear()
        self.hist.hist(data, bins=bins)
        self.hist.set_title(self.label)
        self.hist.set_yscale(yscale)
        self.hist.relim()
        self.hist.autoscale_view()


class PlotPanel:
    def __init__(self, fig, panels, fig_index, label):
        self.fig = fig
        self.label = label
        self.plot = fig.add_subplot(*panels, fig_index, )
        self.plot.set_title(self.label)

    def update(self, sequence, label=''):
        self.plot.set_title(self.label)
        self.plot.plot(sequence, label=label)
        self.plot.relim()
        self.plot.autoscale_view()

    def reset(self):
        self.plot.clear()


class LiveLine:
    def __init__(self, fig, panels, fig_index, label, length=1000):
        self.fig = fig
        self.label = label
        self.live = fig.add_subplot(*panels, fig_index, )
        self.live.set_title(self.label)
        self.rew_live_length = length
        self.dq = deque(maxlen=self.rew_live_length)

    def update(self, data):
        self.live.clear()
        self.dq.append(data)
        y = np.array(self.dq)
        self.live.set_title(self.label)
        self.live.plot(y)
        self.live.relim()
        self.live.autoscale_view()

    def draw(self):
        self.live.clear()
        y = np.array(self.dq)
        self.live.set_title(self.label)
        self.live.plot(y)
        self.live.relim()
        self.live.autoscale_view()

    def reset(self):
        self.live.clear()
        self.dq = deque(maxlen=self.rew_live_length)


class MultiLiveLine:
    def __init__(self, fig, panels, fig_index, label, length=1000):
        self.fig = fig
        self.label = label
        self.live = fig.add_subplot(*panels, fig_index, )
        self.live.set_title(self.label)
        self.rew_live_length = length
        self.plots = {}

    def update(self, **kwargs):
        self.live.clear()
        for name, value in kwargs.items():
            if name not in self.plots:
                self.plots[name] = deque(maxlen=self.rew_live_length)
            self.plots[name].append(value)
            y = np.array(self.plots[name])
            self.live.plot(y, label=name)
        self.live.relim()
        self.live.autoscale_view()
        self.live.set_title(self.label)
        self.live.legend()

    def reset(self):
        self.live.clear()
        self.dq = {}


class Viz:
    def __init__(self, args, figsize=(12, 16), panel_grid=(4, 2), window_title=None):
        plt.ion()
        self.fig = plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k', )
        plt.subplots_adjust(hspace=0.5)
        if window_title is not None:
            self.fig.canvas.set_window_title(window_title)
        self.args = args
        self.panel_grid = panel_grid
        self.current_panel = 1
        self.panels = {}

    def next_panel(self):
        current_panel = self.current_panel
        self.current_panel += 1
        return current_panel

    def draw(self):
        self.fig.canvas.draw()


class VizBoard2(Viz, EnvObserver):
    def __init__(self, args, figsize=(12, 8), panel_grid=(4, 2), window_title=None):
        super().__init__(args, figsize=figsize, panel_grid=panel_grid, window_title=window_title)
        self.panels['env_reward_sum'] = MultiLiveLine(self.fig, self.panel_grid, self.next_panel(), label='sum reward')
        self.panels['env_reward_mean'] = MultiLiveLine(self.fig, self.panel_grid, self.next_panel(), label='mean reward')

        self.panels['episode_len'] = MultiLiveLine(self.fig, self.panel_grid, self.next_panel(), label='episode length')
        self.panels['policy_grad'] = LiveLine(self.fig, self.panel_grid, self.next_panel(), label='policy gradient')
        self.returns = []
        self.model_returns = []
        self.episode_length = 0

    def update_returns(self):
        returns = np.array(self.returns)
        if len(self.model_returns) > 0:
            model_returns = np.array(self.model_returns)
            self.panels['env_reward_sum'].update(model=model_returns.sum().item(), env=returns.sum().item())
            self.panels['env_reward_mean'].update(model=model_returns.mean().item(), env=returns.mean().item())
        else:
            self.panels['env_reward_sum'].update(env=returns.sum().item())
            self.panels['env_reward_mean'].update(env=returns.mean().item())

    def sample_grad_norm(self, model, sample=0.01):
        if random.random() < sample:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.panels['policy_grad'].update(total_norm)
            self.draw()

    def step(self, state, action, reward, done, info, **kwargs):
        self.episode_length += 1
        self.returns.append(reward)
        if 'model_reward' in kwargs:
            self.model_returns.append(kwargs['model_reward'])

    def done(self):
        self.panels['episode_len'].update(length=self.episode_length)
        self.episode_length = 0
        self.update_returns()
        self.draw()
        self.returns = []


class VizBoard:
    def __init__(self, args, window_title=None):
        plt.ion()
        self.fig = plt.figure(num=None, figsize=(24, 16), dpi=80, facecolor='w', edgecolor='k', )
        if window_title:
            self.fig.canvas.set_window_title(window_title)
        self.args = args
        self.current_panel = 1

        panels = (5, 8)
        self.rew_plot = LiveLine(self.fig, panels, self._next_panel, label='sum reward')
        self.rew_mean_plot = LiveLine(self.fig, panels, self._next_panel, label='mean reward')
        self.steps = LiveLine(self.fig, panels, self._next_panel, label='episode length')

        self.dynamics_hist = HistogramPanel(self.fig, panels, self._next_panel, label='dynamics')
        self.rew_hist = HistogramPanel(self.fig, panels, self._next_panel, label='reward')
        self.prew_hist = HistogramPanel(self.fig, panels, self._next_panel, label='predicted_reward')
        self.raw_pcont_hist = HistogramPanel(self.fig, panels, self._next_panel, label='measured pcont')
        self.est_pcont_hist = HistogramPanel(self.fig, panels, self._next_panel, label='est pcont')
        self.value_hist = HistogramPanel(self.fig, panels, self._next_panel, label='value')
        self.sampled_value_hist = HistogramPanel(self.fig, panels, self._next_panel, label='sampled value')

        self.policy_grad_norm = LiveLine(self.fig, panels, self._next_panel, label='policy gradient')
        self.live_policy_entropy = PlotPanel(self.fig, panels, fig_index=self._next_panel, label='entropy')

        self.live_value = PlotPanel(self.fig, panels, fig_index=self._next_panel, label='episode value')
        self.live_pcont = PlotPanel(self.fig, panels, fig_index=self._next_panel, label='pcont')
        self.exp_rew_vs_actual = PlotPanel(self.fig, panels, fig_index=self._next_panel, label='dyn next_step reward')
        self.live_reward = PlotPanel(self.fig, panels, fig_index=self._next_panel, label='epi reward')
        self.live_dynamics = PlotPanel(self.fig, panels, fig_index=self._next_panel, label='epi dyn prob')
        self.live_dynamics_entropy = PlotPanel(self.fig, panels, fig_index=self._next_panel, label='epi dyn ent')
        self.live_dynamics_reward_prob = PlotPanel(self.fig, panels, fig_index=self._next_panel, label='epi dyn reward prob')
        self.live_dynamics_done_prob = PlotPanel(self.fig, panels, fig_index=self._next_panel, label='epi dyn done prob')
        self.live_dynamics_contact_prob = PlotPanel(self.fig, panels, fig_index=self._next_panel,
                                                    label='epi dyn contact prob')

        self.fig.canvas.draw()
        self.samples_in_histogram = 500

    @property
    def _next_panel(self):
        current_panel = self.current_panel
        self.current_panel += 1
        return current_panel

    def plot_rewards_histogram(self, b, R):
        with torch.no_grad():
            if len(b.index) < self.samples_in_histogram:
                index = b.index
            else:
                index = random.sample(b.index,self.samples_in_histogram)
            r = np.concatenate([b.trajectories[t][i].reward for t, i in index])
            s = np.stack([b.trajectories[t][i].state for t, i in index])
            R.eval()
            pr = R(torch.from_numpy(s).to(device=self.args.device))
            pr = pr.cpu().detach().numpy()
            R.train()

            self.rew_hist.update(r, bins=100, yscale='log')
            self.prew_hist.update(pr, bins=100, yscale='log')
            self.fig.canvas.draw()

    def update_rewards(self, reward):
        """ accepts a list of rewards at each step"""
        self.rew_plot.update(sum(reward))
        self.rew_mean_plot.update(mean(reward))
        self.steps.update(len(reward))
        self.fig.canvas.draw()

    def _draw_samples(self, b):
        if len(b.index) < self.samples_in_histogram:
            index = b.index
        else:
            index = random.sample(b.index, self.samples_in_histogram)
        return index

    def update_pcont(self, b, pcont):
        with torch.no_grad():
            index = self._draw_samples(b)
            s = np.stack([b.trajectories[t][i].state for t, i in index])
            pcont_raw = np.stack([b.trajectories[t][i].pcont for t, i in index])
            pred_pcont = pcont(torch.from_numpy(s).to(device=self.args.device))
            self.raw_pcont_hist.update(pcont_raw, bins=100, yscale='log')
            self.est_pcont_hist.update(pred_pcont.cpu().detach().numpy(), bins=100, yscale='log')

    def update_value(self, b, value):
        with torch.no_grad():
            index = self._draw_samples(b)
            s = np.stack([b.trajectories[t][i].state for t, i in index])
            pred_value = value(torch.from_numpy(s).to(device=self.args.device))
            self.value_hist.update(pred_value.cpu().detach().numpy(), bins=100, yscale='log')

    def update_trajectory_plots(self, value, R, pcont, T, b, trajectory_id, entropy):
        with torch.no_grad():
            self.live_value.reset(), self.live_pcont.reset(), self.exp_rew_vs_actual.reset()
            self.live_reward.reset(), self.live_dynamics.reset(), self.live_policy_entropy.reset()
            self.live_dynamics_entropy.reset(), self.live_dynamics_reward_prob.reset()
            self.live_dynamics_done_prob.reset(), self.live_dynamics_contact_prob.reset()

            s = np.stack(i.state for i in b.trajectories[trajectory_id])
            a = np.stack(i.action for i in b.trajectories[trajectory_id])
            r = np.stack(i.reward for i in b.trajectories[trajectory_id])
            p = np.stack(i.pcont for i in b.trajectories[trajectory_id])
            s = torch.from_numpy(s).to(device=self.args.device)
            a = torch.from_numpy(a).to(device=self.args.device)

            N, S = s.size()

            pc = pcont(s).squeeze().cpu().numpy()

            def next_state_lstm(s, a):
                sa = torch.cat((s, a), dim=1)
                pred_next_dist, hx = T(sa.unsqueeze(1))
                next_state = pred_next_dist.loc.reshape(N, S)
                return next_state

            def next_state_ode(s, a):
                t = torch.linspace(0, 1, 2)
                T.mode = 'learn_dynamics'
                T.h = torch.stack([a, a])
                predicted_trajectory = odeint(T, s, t)
                return predicted_trajectory[1:]

            #next_state = next_state_lstm(s, a)

            next_state = next_state_ode(s, a)
            pr_next = R(next_state).squeeze().cpu().numpy()
            pr = R(s).squeeze().cpu().numpy()

            # update pcont
            self.live_pcont.update(pc)
            self.live_pcont.update(p)

            # update expected reward
            self.exp_rew_vs_actual.update(pr)
            self.exp_rew_vs_actual.update(r)
            self.exp_rew_vs_actual.update(pr_next * pc)

            # update predicted vs recieved
            self.live_reward.update(pr, 'predicted')
            self.live_reward.update(r, 'received')
            self.live_reward.update(pr * pc)

            self.update_episode_value(value, s)
            #self.update_episode_dynamics(T, s, a)
            self.update_policy_entropy(entropy)
            self.fig.canvas.draw()

    def update_episode_value(self, value, s):
        with torch.no_grad():
            v = value(s).squeeze().cpu().numpy()
            self.live_value.update(v)

    def update_episode_dynamics(self, T, s, a):
        with torch.no_grad():
            sa = torch.cat((s[:-1], a[:-1]), dim=1)
            dist, hx = T(sa.unsqueeze(1))
            N, S = s.size()
            p = torch.exp(dist.log_prob(s[1:].unsqueeze((1)))).reshape(N-1, S)
            mean_p = p.mean(1).squeeze().cpu().numpy()
            #done_p = p[:, -1].squeeze().cpu().numpy()
            #reward_p = p[:, -2].squeeze().cpu().numpy()
            #contact_p = p[:, 20:26].mean(1).squeeze().cpu().numpy()
            entropy = dist.entropy().mean(1).squeeze().cpu().numpy()
            self.live_dynamics.update(mean_p)
            self.live_dynamics_entropy.update(entropy)
            #self.live_dynamics_reward_prob.update(reward_p)
            #self.live_dynamics_done_prob.update(done_p)
            #self.live_dynamics_contact_prob.update(contact_p)

    def sample_grad_norm(self, model, sample=0.01):
        if random.random() < sample:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.policy_grad_norm.update(total_norm)
            self.fig.canvas.draw()

    def update_dynamics(self, b, T):
        with torch.no_grad():
            if len(b.index) < self.samples_in_histogram:
                index = b.index
            else:
                index = random.sample(b.index, self.samples_in_histogram)
            non_terminals = []
            for t, i in index:
                if not b.trajectories[t][i].done:
                    non_terminals.append((t, i))
            s = np.stack([b.trajectories[t][i].state for t, i in non_terminals])
            a = np.stack([b.trajectories[t][i].action for t, i in non_terminals])
            s = torch.from_numpy(s).to(device=self.args.device)
            a = torch.from_numpy(a).to(device=self.args.device)
            sa = torch.cat((s, a), dim=1)
            next_s = np.stack([b.trajectories[t][i+1].state for t, i in non_terminals])
            pred_next_dist, hx = T(sa.unsqueeze(0))
            prob_next = pred_next_dist.log_prob(torch.from_numpy(next_s).to(device=self.args.device)).exp()


            prob_next = prob_next.cpu().detach().numpy()
            prob_next = prob_next.flatten()
            self.dynamics_hist.update(prob_next, bins=100)

    def update_sampled_values(self, values):
        values = np.concatenate(tuple(values))
        self.sampled_value_hist.update(values, yscale='log')

    def update_policy_entropy(self, entropy):
        self.live_policy_entropy.update(entropy, 'entropy')


def make_map(type_name, size):
    state_map = {}
    for i in range(size):
        state_map[f'{type_name} {i}'] = i
    return state_map

def _np(x):
    return x.detach().cpu().numpy()


class VizTransition:
    def __init__(self, state_dims, action_dims, figsize=(12, 12), layout=(8, 2), title='Dynamics',
                 state_map=None, action_map=None):
        """  will draw trajectories for states and actions
        to configure pass a dict that maps index to name

        state_map =  {
            'x_pos': 0,
            'y_pos': 1,
            'x_vel' : 2,
            'y_vel' : 3,
        }

        else it will try to create a layout for state_dims and action_dims

        update the panel by using the update method

        """
        plt.ion()

        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.set_window_title(title)

        self.layout = layout
        subplots = self.layout[0] * self.layout[1]
        if subplots < (state_dims + action_dims):
            int_sqrt = math.ceil(math.sqrt(subplots))
            self.layout = (int_sqrt, int_sqrt)

        self.next_panel = 1
        self.panels = {}

        self.state_map = make_map('state', state_dims) if state_map is None else state_map
        self.action_map = make_map('action', action_dims) if action_map is None else action_map

        for key in self.state_map:
            self.add_subplot(key)
        for key in self.action_map:
            self.add_subplot(key)

    def add_subplot(self, key):
        self.panels[key] = self.fig.add_subplot(*self.layout, self.next_panel, )
        self.next_panel += 1

    def update_panel(self, key, i, t, trajectories, prediction=None):
        panel = self.panels[key]
        panel.clear()
        panel.set_title(key)
        panel.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        panel.plot(_np(t), _np(trajectories[1:, 0, i]), label='trajectory')
        if prediction is not None:
            panel.plot(_np(t), _np(prediction[:, 0, i]), label='prediction')
        panel.legend()

    def update(self, trajectories, prediction, t=None):
        """
        call this method to update the panel
        :param t: array of trajectory
        :param trajectories: the batch, a Namespace of tensors returned by collate_pad_2
        :param prediction: a tensor of states, L, N S
        :return:
        """

        L, N, S = trajectories.state.shape

        t = torch.linspace(0, L, L-1) if t is None else t

        for key, i in self.state_map.items():
            self.update_panel(key, i, t, trajectories.state, prediction)

        for key , i in self.action_map.items():
            self.update_panel(key, i, t, trajectories.action)

        self.fig.canvas.draw()