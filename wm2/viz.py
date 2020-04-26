import curses
from collections import OrderedDict

from matplotlib import pyplot as plt
from time import sleep

from math import floor

import cv2
import numpy as np
import torch

from functional import multivariate_diag_gaussian, multivariate_gaussian


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

        curses.init_pair(1, 0, -1)  # slot text color
        curses.init_pair(2, 139, -1)  # status bar color

        self.height, self.width = self.stdscr.getmaxyx()

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

    def update_table(self, table, h=0, title=None):
        assert len(table.shape) == 2
        if title is not None:
            self._write_row(title, h=h)
            h = h + 1
        for i in range(table.shape[0]):
            table_str = np.array2string(table[i], max_line_width=self.width)
            self._write_row(table_str, i + h)
        self.refresh()


class LineViz:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)
        s = torch.linspace(-2.5, 2.5, 20).view(-1, 1)
        z = torch.zeros(20, 1)
        self.l_actions, = self.ax1.plot(s, z, 'b-', label='policy(state)')
        self.l_rewards, = self.ax2.plot(s, z, 'b-', label='reward(state)')
        self.l_next_state_0_2, = self.ax3.plot(s, z, 'b-', label='T(state,0.2)')
        self.l_next_state_minus_0_2, = self.ax3.plot(s, z, 'r-', label='T(state,-0.2)')
        self.l_value, = self.ax4.plot(s, z, 'b-', label='value(state)')
        self.ax1.legend(), self.ax2.legend(), self.ax3.legend(), self.ax4.legend()

    def update(self, next_state_0_2, next_state_minus_0_2, a, r):

        self.l_actions.set_ydata(a.detach().cpu().numpy())
        self.l_rewards.set_ydata(r.detach().cpu().numpy())
        self.l_next_state_0_2.set_ydata(next_state_0_2.detach().cpu().numpy())
        self.l_next_state_minus_0_2.set_ydata(next_state_minus_0_2.detach().cpu().numpy())
        self.l_value.set_ydata(v.detach().cpu().numpy())

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.ax4.relim()
        self.ax4.autoscale_view()
        self.fig.canvas.draw()