from collections import deque
from statistics import mean
from types import SimpleNamespace
from pathlib import Path
import sys
from datetime import datetime
from math import floor

import wandb
import torch
from tqdm import tqdm


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

    def update_train_loss_and_checkpoint(self, loss, models=None, epoch=None, optimizer=None):
        self.test = []
        self.loss_move_ave.append(loss.item())
        self.train_loss = mean(list(self.loss_move_ave))
        self.bar.update(self.batch_size)
        wandb.log({f'{self.label}_train_loss': loss.item()})
        self.bar.set_description(f'{self.label} train_loss: {self.train_loss:.6f} test_loss: {self.test_loss:.6f}')
        if models is not None and self.checkpoint_cooldown:
            save = {}
            save['epoch'] = epoch
            save['optimizer_state_dict'] = optimizer.state_dict()
            for name, model in models.items():
                save[name] = model.state_dict()
            torch.save(save, str(Path(wandb.run.dir) / Path(f'{self.label}_checkpoint.pt')))

    def update_test_loss_and_save_model(self, loss, models=None):
        self.test.append(loss.item())
        self.test_loss = mean(list(self.test))
        wandb.log({f'{self.label}_test_loss': loss.item()})
        self.bar.set_description(f'{self.label} train_loss: {self.train_loss:.6f} test_loss: {self.test_loss:.6f}')
        if models is not None:
            if loss.item() < self.best_loss:
                save = {}
                self.best_loss = loss.item()
                for name, model in models.items():
                    save['loss'] = loss.item()
                    save[name] = model.state_dict()
                torch.save(save, str(Path(wandb.run.dir) / Path(f'{self.label}_best.pt')))

    def close(self):
        self.bar.close()

    @staticmethod
    def best_state_dict(wandb_run_dir, label):
        f = str(Path(wandb_run_dir) / Path(f'{label}_best.pt'))
        return torch.load(f)

    @staticmethod
    def checkpoint(wandb_run_dir, label):
        f = str(Path(wandb_run_dir) / Path(f'{label}_checkpoint.pt'))
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


