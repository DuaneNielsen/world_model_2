from collections import deque
from statistics import mean
from types import SimpleNamespace
from pathlib import Path
import sys
from datetime import datetime
from math import floor

import wandb
import cv2
import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from collections import Iterable


class TensorNamespace(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to(self, device):
        for key in vars(self):
            vars(self)[key] = vars(self)[key].to(device)
        return self


class Pbar:
    def __init__(self, epochs, train_len, batch_size, label, train_depth=None, checkpoint_secs=60):
        """

        :param epochs: number of epochs to train
        :param train_len: length of the training dataset (number of items)
        :param batch_size: items per batch
        :param label: a label to display on the progress bar
        :param train_depth: the previous number of training losses to average to display the training loss
        :param checkpoint_secs: the number of seconds before saving a checkpoint
        if not set will default to 10 or the number of minibatches per epoch, whichever is lower
        """
        self.bar = tqdm(total=epochs * train_len)
        self.label = label
        depth = train_depth if train_depth is not None else min(train_len//batch_size, 10)
        self.loss_move_ave = deque(maxlen=depth)
        self.test = []
        self.test_loss = 0.0
        self.train_loss = 0.0
        self.batch_size = batch_size
        self.best_loss = sys.float_info.max
        self.checkpoint_cooldown = Cooldown(checkpoint_secs)


    def update_train_loss_and_checkpoint(self, loss, model=None, epoch=None, optimizer=None):
        self.test = []
        self.loss_move_ave.append(loss.item())
        self.train_loss = mean(list(self.loss_move_ave))
        self.bar.update(self.batch_size)
        wandb.log({f'{self.label}_train_loss': loss.item()})
        self.bar.set_description(f'{self.label} train_loss: {self.train_loss:.6f} test_loss: {self.test_loss:.6f}')
        if model is not None and self.checkpoint_cooldown:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(Path(wandb.run.dir) / Path(f'{self.label}_checkpoint.pt')))

    def update_test_loss_and_save_model(self, loss, model=None):
        self.test.append(loss.item())
        self.test_loss = mean(list(self.test))
        wandb.log({f'{self.label}_test_loss': loss.item()})
        self.bar.set_description(f'{self.label} train_loss: {self.train_loss:.6f} test_loss: {self.test_loss:.6f}')
        if model is not None:
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                torch.save(model.state_dict(), str(Path(wandb.run.dir) / Path(f'{self.label}_best.pt')))

    def close(self):
        self.bar.close()

    @staticmethod
    def best_state_dict(wandb_run_dir, label):
        f = str(Path(wandb_run_dir) / Path(f'{label}_best.pt'))
        return torch.load(f)

    @staticmethod
    def checkpoint(wandb_run_dir, label):
        f = str(Path(wandb_run_dir) / Path(f'{label}_best.pt'))
        return torch.load(f)


class Cooldown:
    def __init__(self, secs=None):
        """
        Cooldown timer. to use, just construct and call it with the number of seconds you want to wait
        default is 1 minute, first time it returns true
        """
        self.last_cooldown = 0
        self.default_cooldown = 60 if secs is None else secs

    def __call__(self, secs=None):
        secs = self.default_cooldown if secs is None else secs
        now = floor(datetime.now().timestamp())
        run_time = now - self.last_cooldown
        expired = run_time > secs
        if expired:
            self.last_cooldown = now
        return expired


class SARI:
    def __init__(self, state, action, reward, done, info):
        self.state = state
        self.action = action
        self.reward = np.array([reward], dtype=state.dtype)
        self.has_reward = reward != 0
        self.done = done
        self.info = info


def one_hot(a, max, dtype=np.float32):
    hot = np.zeros(max + 1, dtype=dtype)
    hot[a] = 1.0
    return hot


def split(data, train_len, test_len):
    total_len = train_len + test_len
    train = Subset(data, range(0, train_len))
    test = Subset(data, range(train_len, total_len))
    return train, test


def squared_diff(h, len):
    """
    keypoints, K
    :param h: K, 1
    :param len: integer
    :return: heightmap
    """
    ls = np.linspace(0, 1, len, dtype=h.dtype)
    ls = np.expand_dims(ls, axis=0).repeat(repeats=h.shape[0], axis=0)
    hm = np.expand_dims(h, axis=0).repeat(repeats=len, axis=0).T
    hm = ls - hm
    return hm ** 2


def gaussian_like_function(kp, height, width, sigma=0.1, eps=1e-6):
    hm = squared_diff(kp[:, 0], height)
    wm = squared_diff(kp[:, 1], width)
    hm = hm[:, :, np.newaxis]
    hm = np.repeat(hm, repeats=width, axis=2)
    wm = wm[:, np.newaxis, :]
    wm = np.repeat(wm, repeats=height, axis=1)
    gm = - np.sqrt(hm + wm + eps) / (2 * sigma ** 2)
    gm = np.exp(gm)
    return gm


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