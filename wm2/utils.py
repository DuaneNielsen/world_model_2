from collections import deque
from statistics import mean
from types import SimpleNamespace

import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm


class TensorNamespace(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to(self, device):
        for key in vars(self):
            vars(self)[key] = vars(self)[key].to(device)
        return self


class Pbar:
    def __init__(self, epochs, train_len, batch_size, label, train_depth=None):
        """

        :param epochs: number of epochs to train
        :param train_len: length of the training dataset (number of items)
        :param batch_size: items per batch
        :param label: a label to display on the progress bar
        :param train_depth: the previous number of training losses to average to display the training loss
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

    def update_test_loss(self, loss):
        self.test.append(loss.item())
        self.test_loss = mean(list(self.test))
        self.bar.set_description(f'{self.label} train_loss: {self.train_loss:.6f} test_loss: {self.test_loss:.6f}')

    def update_train_loss(self, loss):
        self.test = []
        self.loss_move_ave.append(loss.item())
        self.train_loss = mean(list(self.loss_move_ave))
        self.bar.update(self.batch_size)
        self.bar.set_description(f'{self.label} train_loss: {self.train_loss:.6f} test_loss: {self.test_loss:.6f}')

    def close(self):
        self.bar.close()


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