from types import SimpleNamespace
from collections import deque, OrderedDict
from statistics import mean
from random import random
import curses

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import wandb
import gym

import matplotlib
import matplotlib.pyplot as plt

from distributions import TanhTransformedGaussian, ScaledTanhTransformedGaussian
from wm2.data.datasets import Buffer, SARDataset, SARNextDataset, SDDataset
from wm2.utils import Pbar
from data.utils import pad_collate_2
import wm2.utils


class LinEnv:
    def __init__(self):
        self.pos = np.array([0.0], dtype=np.float32)
        self.step_count = 0

    def reset(self):
        self.pos = np.array([0.0], dtype=np.float32)
        self.step_count = 0
        return self.pos.copy()

    def step(self, action):
        self.step_count += 1
        self.pos += action
        pos = self.pos.copy()

        if pos >= 2.0:
            return pos, 1.0, True
        elif self.step_count > 10:
            return pos, 0.0, True
        elif pos <= -2.0:
            return pos, -1.0, True
        else:
            return pos, 0.0, False


def policy_prepro(state):
    return torch.tensor(state).float().to(args.device)


def buffer_prepro(state):
    return state.astype(np.float32)


def action_prepro(action):
    return np.array([action.item()], dtype=np.float32)


def reward_prepro(reward):
    return np.array([reward], dtype=np.float32)


def random_policy(state):
    return TanhTransformedGaussian(0.0, 0.5)


def gather_experience(buff, episode, env, policy, eps=0.0, eps_policy=None, render=True):
    with torch.no_grad():
        # gather new experience
        episode_reward = 0.0
        state, reward, done = env.reset(), 0.0, False
        if random() >= eps:
            action = policy(policy_prepro(state).unsqueeze(0)).rsample()
        else:
            action = eps_policy(policy_prepro(state).unsqueeze(0)).rsample()
        action = action_prepro(action)
        buff.append(episode, buffer_prepro(state), action, reward_prepro(reward), done, None)
        episode_reward += reward
        if render:
            env.render()
        while not done:
            state, reward, done, info = env.step(action)
            episode_reward += reward
            if random() >= eps:
                action = policy(policy_prepro(state).unsqueeze(0)).rsample()
            else:
                action = eps_policy(policy_prepro(state).unsqueeze(0)).rsample()
            action = action_prepro(action)
            buff.append(episode, buffer_prepro(state), action, reward_prepro(reward), done, None)
            if render:
                env.render()
    return buff, episode_reward


def gather_seed_episodes(env, seed_episodes):
    buff = Buffer()
    for episode in range(seed_episodes):
        gather_experience(buff, episode, env, random_policy, render=False)
    return buff


class MLP(nn.Module):
    def __init__(self, layers, nonlin=None):
        super().__init__()
        in_dims = layers[0]
        net = []
        # the use of eval here is a security bug
        nonlin = nn.ELU if nonlin is None else eval(nonlin)
        for hidden in layers[1:-1]:
            net += [nn.Linear(in_dims, hidden)]
            net += [nonlin()]
            in_dims = hidden
        net += [nn.Linear(in_dims, layers[-1], bias=False)]

        self.mlp = nn.Sequential(*net)

    def forward(self, inp):
        return self.mlp(inp)


class Policy(nn.Module):
    def __init__(self, layers, min=-1.0, max=1.0):
        super().__init__()
        self.mu = MLP(layers)
        # self.scale = nn.Linear(state_dims, 1, bias=False)
        self.min = min
        self.max = max

    def forward(self, state):
        mu = self.mu(state)
        # scale = torch.sigmoid(self.scale(state))
        return ScaledTanhTransformedGaussian(mu, 0.2, min=self.min, max=self.max)


class TransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, dropout=dropout)
        self.outnet = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, inp, hx=None):
        output, hx = self.lstm(inp, hx)
        output = self.outnet(output)
        return output, hx


def reward_mask_f(state, reward, action):
    r = np.concatenate(reward)
    nonzero = r != 0
    p = np.ones_like(r)
    p = p / (r.shape[0] - nonzero.sum())
    p = p * ~nonzero
    i = np.random.choice(r.shape[0], nonzero.sum(), p=p)
    nonzero[i] = True
    return nonzero[:, np.newaxis]


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


def main(args):
    # curses
    scr = Curses()

    # monitoring
    recent_reward = deque(maxlen=20)
    wandb.gym.monitor()
    imagine_log_cooldown = wm2.utils.Cooldown(secs=30)
    transition_log_cooldown = wm2.utils.Cooldown(secs=30)

    # visualization
    # plt.ion()
    # fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax4 = fig.add_subplot(224)
    # s = torch.linspace(-2.5, 2.5, 20).view(-1, 1)
    # z = torch.zeros(20, 1)
    # l_actions, = ax1.plot(s, z, 'b-', label='policy(state)')
    # l_rewards, = ax2.plot(s, z, 'b-', label='reward(state)')
    # l_next_state_0_2, = ax3.plot(s, z, 'b-', label='T(state,0.2)')
    # l_next_state_minus_0_2, = ax3.plot(s, z, 'r-', label='T(state,-0.2)')
    # l_value, = ax4.plot(s, z, 'b-', label='value(state)')
    # ax1.legend(), ax2.legend(), ax3.legend(), ax4.legend()

    # environment
    # env = LinEnv()
    env = gym.make('Pendulum-v0')
    # env = gym.make('MountainCarContinuous-v0')
    train_buff = gather_seed_episodes(env, args.seed_episodes)
    test_buff = gather_seed_episodes(env, args.seed_episodes)
    train_episode, test_episode = args.seed_episodes, args.seed_episodes

    eps = 0.05

    # # representation model
    # REPR = TransitionModel(input_dim=args.state_dims + args.action_dims,
    #                        hidden_dim=3, output_dim=args.state_dims,
    #                        layers=args.transition_layers).to(args.device)

    # policy model
    policy = Policy(layers=[args.state_dims, *args.policy_hidden_dims, 1], min=-2.0, max=2.0).to(args.device)
    policy_optim = Adam(policy.parameters(), lr=args.lr)

    # value model
    # value = nn.Linear(state_dims, 1)
    value = MLP([args.state_dims, *args.value_hidden_dims, 1], nonlin=args.nonlin).to(args.device)
    value_optim = Adam(value.parameters(), lr=args.lr)

    # transition model
    # T = nn.LSTM(input_size=state_dims + action_dims, hidden_size=state_dims, num_layers=2)
    T = TransitionModel(input_dim=args.state_dims + args.action_dims,
                        hidden_dim=args.transition_hidden_dim, output_dim=args.state_dims,
                        layers=args.transition_layers).to(args.device)
    T_optim = Adam(T.parameters(), lr=args.lr)

    # reward model
    R = MLP([args.state_dims, *args.reward_hidden_dims, 1], nonlin=args.nonlin).to(args.device)
    # R = nn.Linear(state_dims, 1)
    R_optim = Adam(R.parameters(), lr=args.lr)

    # terminal state model
    # D = nn.Linear(state_dims, 1).to(args.device)
    # D_optim = Adam(D.parameters(), lr=args.lr)

    converged = False

    scr.clear()

    while not converged:

        for c in range(args.collect_interval):

            scr.update_progressbar(c)
            scr.update_slot('wandb', f'{wandb.run.name} {wandb.run.project} {wandb.run.id}')

            # Dynamics learning
            train, test = SARNextDataset(train_buff, mask_f=None), SARNextDataset(test_buff, mask_f=None)
            train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            # pbar = Pbar(items_to_process=args.trajectories_per_pass, train_len=len(train), batch_size=args.batch_size,
            #             label='transition')
            # while pbar.items_processed < args.trajectories_per_pass:

            for _ in range(1):

                # train transition model
                for trajectories in train:
                    input = torch.cat((trajectories.state, trajectories.action), dim=2).to(args.device)
                    T_optim.zero_grad()
                    predicted_state, (h, c) = T(input)
                    loss = ((trajectories.next_state.to(args.device) - predicted_state) ** 2) * trajectories.pad.to(
                        args.device)
                    loss = loss.mean()
                    loss.backward()
                    T_optim.step()
                    scr.update_slot('transition_train', f'Transition training loss {loss.item()}')
                    wandb.log({'transition_train': loss.item()})
                    # pbar.update_train_loss_and_checkpoint(loss, models={'transition': T}, optimizer=T_optim)

                for trajectories in test:
                    input = torch.cat((trajectories.state, trajectories.action), dim=2).to(args.device)
                    predicted_state, (h, c) = T(input)
                    loss = ((trajectories.next_state.to(args.device) - predicted_state) ** 2) * trajectories.pad.to(
                        args.device)
                    loss = loss.mean()
                    scr.update_slot('transition_test', f'Transition test loss  {loss.item()}')
                    wandb.log({'transition_test': loss.item()})
                    if transition_log_cooldown():
                        scr.update_table(trajectories.next_state[10:20, 0, :].detach().cpu().numpy().T, h=10,
                                         title='next_state')
                        scr.update_table(predicted_state[10:20, 0, :].detach().cpu().numpy().T, h=14,
                                         title='predicted')
                        scr.update_table(trajectories.action[10:20, 0, :].detach().cpu().numpy().T, h=16,
                                         title='action')

                    # pbar.update_test_loss_and_save_model(loss, models={'transition': T})
            # pbar.close()

            # Reward learning
            # train, test = SARDataset(train_buff, mask_f=reward_mask_f), SARDataset(test_buff, mask_f=reward_mask_f)
            train, test = SARDataset(train_buff), SARDataset(test_buff)
            train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            test = DataLoader(test, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            # pbar = Pbar(items_to_process=args.trajectories_per_pass, train_len=len(train),
            #             batch_size=args.batch_size, label='reward')
            # while pbar.items_processed < args.trajectories_per_pass:
            for _ in range(1):
                for trajectories in train:
                    R_optim.zero_grad()
                    predicted_reward = R(trajectories.state.to(args.device))
                    loss = (((trajectories.reward.to(args.device) - predicted_reward) * trajectories.pad.to(
                        args.device)) ** 2).mean()
                    # loss = ((trajectory.reward.to(args.device) - predicted_reward) ** 2).mean()
                    loss.backward()
                    R_optim.step()
                    scr.update_slot('reward_train', f'Reward train loss {loss.item()}')
                    wandb.log({'reward_train': loss.item()})
                    # pbar.update_train_loss_and_checkpoint(loss)

                for trajectories in test:
                    predicted_reward = R(trajectories.state.to(args.device))
                    loss = (((trajectories.reward.to(args.device) - predicted_reward) * trajectories.pad.to(
                        args.device)) ** 2).mean()
                    scr.update_slot('reward_test', f'Reward test loss  {loss.item()}')
                    wandb.log({'reward_test': loss.item()})
                    # loss = ((trajectory.reward.to(args.device) - predicted_reward) ** 2).mean()
                    # pbar.update_test_loss_and_save_model(loss)
            # pbar.close()

            # Terminal state learning
            # train, test = SDDataset(train_buff), SDDataset(test_buff)
            # train_weights, test_weights = train.weights(), test.weights()
            # train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
            # test_sampler = WeightedRandomSampler(test_weights, len(test_weights))
            # train = DataLoader(train, batch_size=32, sampler=train_sampler, drop_last=False)
            # test = DataLoader(test, batch_size=32, sampler=test_sampler, drop_last=False)
            # pbar = Pbar(items_to_process=len(train) * 2, train_len=len(train),
            #             batch_size=args.batch_size, label='terminal')
            # while pbar.items_processed < len(train) * 2:
            #
            #     for state, done in train:
            #         D_optim.zero_grad()
            #         predicted_done = D(state)
            #         loss = F.binary_cross_entropy_with_logits(predicted_done, done)
            #         loss.backward()
            #         D_optim.step()
            #         pbar.update_train_loss_and_checkpoint(loss)
            #
            #     for state, done in test:
            #         predicted_done = D(state)
            #         loss = F.binary_cross_entropy_with_logits(predicted_done, done)
            #         pbar.update_test_loss_and_save_model(loss)
            #
            # pbar.close()

            # Behaviour learning
            train = ConcatDataset([SARDataset(train_buff)])
            train = DataLoader(train, batch_size=args.batch_size, collate_fn=pad_collate_2, shuffle=True)
            # pbar = Pbar(items_to_process=args.trajectories_per_pass, train_len=len(train),
            #             batch_size=args.batch_size, label='behavior')

            # while pbar.items_processed < 10:
            for _ in range(1):

                for trajectories in train:

                    # compute the cell values for the sampled trajectory
                    h, c = None, None
                    with torch.no_grad():
                        s_traj, h_traj, c_traj = [], [], []
                        h_shape = args.transition_layers, trajectories.state.size(1), args.transition_hidden_dim
                        h, c = torch.zeros(h_shape, device=args.device), torch.zeros(h_shape, device=args.device)

                        for state, action in zip(trajectories.state, trajectories.action):
                            if action.max() > 2.0 or action.min() < -2.0:
                                print(f'imaginination {action.max()} {action.min()}')
                            h_traj.append(h)
                            c_traj.append(c)
                            s = torch.cat((state, action), dim=1).to(args.device).unsqueeze(0)
                            s_traj.append(s)
                            s, (h, c) = T(s, (h, c))

                        h = torch.cat(h_traj, dim=1)
                        c = torch.cat(c_traj, dim=1)
                        s = torch.cat(s_traj, dim=1)

                    imagine = [s]
                    reward = [R(s[:, :, :-args.action_dims])]
                    # done = [D(trajectory.state.to(args.device))]
                    v = [value(s[:, :, :-args.action_dims])]

                    # imagine forward here
                    for tau in range(args.horizon):
                        state, (h, c) = T(imagine[tau], (h, c))
                        action = policy(state).rsample()
                        reward += [R(state)]
                        # done += [D(state)]
                        v += [value(state)]
                        imagine += [torch.cat((state, action), dim=2)]

                    # VR = torch.mean(torch.stack(reward), dim=0)
                    rstack, vstack = torch.stack(reward), torch.stack(v)
                    H, L, N, S = rstack.shape

                    """ construct matrix in form
                    
                    R0 V1  0  0
                    R0 R1 V2  0
                    R0 R1 R2 V3
                    
                    Where R0, R1, R2 are predicted rewards at timesteps 0, 1, 2
                    and V1, V2, V3 are the predicted future values of states at time 1, 2, 3
                    """

                    # create a H * H matrix filled with rewards (using rstack preserves computation graph)
                    rstack = rstack.repeat(H, 1, 1, 1).reshape(H, H, L, N, S)

                    # replace the upper triangle with zeros
                    r, c = torch.triu_indices(H, H)
                    rstack[r, c] = 0.0

                    # replace diagonal rewards with values
                    rstack[torch.arange(H), torch.arange(H)] = vstack[torch.arange(H)]

                    # clip the top row
                    rstack = rstack[1:, :]

                    # occasionally dump table to screen for analysis
                    if imagine_log_cooldown():
                        rewards = rstack[-1:, :, 0, 0, 0].detach().cpu().numpy()
                        scr.update_table(rewards, title='imagined rewards and values')
                        v = vstack[:, 0, 0, 0].unsqueeze(0).detach().cpu().numpy()
                        scr.update_table(v, h=2)
                        imagined_trajectory = torch.stack(imagine)[:, 0, 0, :].detach().cpu().numpy().T
                        scr.update_table(imagined_trajectory, h=3, title='imagined trajectory')

                        # t = wandb.Table(data=)
                        # wandb.log({'imagination': t})

                    """ reduce the above matrix to values using the formula from the paper in 2 steps
                    first, compute VN for each k by applying discounts to each time-step and compute the expected value
                    """

                    # compute and apply the discount (alternatively can estimate the discount using a probability to terminate function)
                    n = torch.linspace(0.0, H - 1, H, device=args.device)
                    discount = torch.full_like(n, args.discount, device=args.device).pow(n).view(1, -1, 1, 1, 1)
                    rstack = rstack * discount

                    # compute the expectation
                    steps = torch.linspace(2, H, H - 1, device=args.device).view(-1, 1, 1, 1)
                    VNK = rstack.sum(dim=1) / steps

                    """ now we are left with a single column matrix in form
                    VN(k=1)
                    VN(k=2)
                    VN(k=3)
                    
                    combine these into a single value with the V lambda equation
                    
                    VL = (1-lambda) * lambda ^ 0  * VN(k=1) + (1 - lambda) * lambda ^ 1 VN(k=2) + lambda ^ 2 * VN(k=3)
                    
                    Note the lambda terms should sum to 1, or you're doing it wrong
                    """

                    lam = torch.full((VNK.size(0),), args.lam, device=args.device).pow(
                        torch.arange(VNK.size(0), device=args.device)).view(-1, 1, 1, 1)
                    lam[0:-1] = lam[0:-1] * (1 - args.lam)
                    VL = (VNK * lam).sum(0)

                    policy_optim.zero_grad(), value_optim.zero_grad()
                    T_optim.zero_grad(), R_optim.zero_grad()  # , D_optim.zero_grad()
                    # policy_loss = - VR.mean()
                    policy_loss = -VL.mean()
                    policy_loss.backward()
                    clip_grad_norm_(parameters=policy.parameters(), max_norm=100.0)
                    policy_optim.step()
                    scr.update_slot('policy_loss', f'Policy loss  {policy_loss.item()}')
                    wandb.log({'policy_loss': policy_loss.item()})

                    # regress against tau ie: the initial estimated value...
                    policy_optim.zero_grad(), value_optim.zero_grad()
                    T_optim.zero_grad(), R_optim.zero_grad()  # , D_optim.zero_grad()

                    VN = VL.detach().reshape(L * N, -1)
                    values = value(trajectories.state.reshape(L * N, -1).to(args.device))
                    value_loss = ((VN - values) ** 2).mean() / 2
                    value_loss.backward()
                    clip_grad_norm_(parameters=value.parameters(), max_norm=100.0)
                    value_optim.step()
                    scr.update_slot('value_loss', f'Value loss  {value_loss.item()}')
                    wandb.log({'value_loss': value_loss.item()})

                    # pbar.update_train_loss_and_checkpoint(policy_loss, models={'policy': policy},
                    #                                                          optimizer=policy_optim)

            # pbar.close()

        for _ in range(3):
            train_buff, reward = gather_experience(train_buff, train_episode, env, policy,
                                                  eps=eps, eps_policy=random_policy)
            train_episode += 1

            wandb.log({'reward': reward})

            recent_reward.append(reward)
            scr.update_slot('eps', f'EPS: {eps}')
            rr = ''
            for reward in recent_reward:
                rr += f' {reward:.5f},'
            scr.update_slot('recent_rewards', 'Recent rewards: ' + rr)

        # a = policy.mu(s)
        # r = R(s)
        # v = value(s)
        #
        # s_0_2 = torch.cat((s.view(1, -1, 1), torch.full((1, 20, 1), 0.2)), dim=2)
        # s_minus_0_2 = torch.cat((s.view(1, -1, 1), torch.full((1, 20, 1), -0.2)), dim=2)
        # next_state_0_2, hidden = T(s_0_2)
        # next_state_minus_0_2, hidden = T(s_minus_0_2)
        #
        # l_actions.set_ydata(a.detach().cpu().numpy())
        # l_rewards.set_ydata(r.detach().cpu().numpy())
        # l_next_state_0_2.set_ydata(next_state_0_2.detach().cpu().numpy())
        # l_next_state_minus_0_2.set_ydata(next_state_minus_0_2.detach().cpu().numpy())
        # l_value.set_ydata(v.detach().cpu().numpy())
        #
        # ax1.relim()
        # ax1.autoscale_view()
        # ax2.relim()
        # ax2.autoscale_view()
        # ax3.relim()
        # ax3.autoscale_view()
        # ax4.relim()
        # ax4.autoscale_view()
        #
        # fig.canvas.draw()

        if random() < 1.1:
            test_buff, reward = gather_experience(test_buff, test_episode, env, policy,
                                                  eps=eps, eps_policy=random_policy)
            test_episode += 1

        # boost or decrease exploration if no reward
        if mean(recent_reward) == 0 and eps < 0.5:
            eps += 0.05
        if mean(recent_reward) > 0.5 and eps > 0.05:
            eps -= 0.05

        converged = False


if __name__ == '__main__':
    args = {'seed_episodes': 40,
            'collect_interval': 10,
            'batch_size': 1,
            'device': 'cuda:0',
            'horizon': 15,
            'discount': 0.99,
            'lam': 0.95,
            'lr': 1e-3,
            'state_dims': 3,
            'policy_hidden_dims': [300],
            'value_hidden_dims': [300],
            'reward_hidden_dims': [300, 300],
            'nonlin': 'nn.ELU',
            'action_dims': 1,
            'transition_layers': 2,
            'transition_hidden_dim': 32
            }

    wandb.init(config=args)
    args = SimpleNamespace(**args)

    curses.wrapper(main(args))
