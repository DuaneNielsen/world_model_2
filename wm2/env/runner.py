import time
import torch
from collections import OrderedDict


class EnvObserver:
    def reset(self):
        """ called before environment reset"""
        pass

    def step(self, state, action, reward, done, info, **kwargs):
        """ called each environment step """
        pass

    def done(self):
        """ called when episode ends """
        pass


class StepFilter:
    """
    Step filters are used to preprocess steps before handing them to observers
    """
    def __call__(self, state, action, reward, done, info, **kwargs):
        return state, action, reward, done, info, kwargs


class RewardFilter(StepFilter):
    def __init__(self, state_prepro, R, device):
        self.state_prepro = state_prepro
        self.R = R
        self.device = device

    def __call__(self, state, action, reward, done, info, **kwargs):
        r = self.R(self.state_prepro(state, self.device))
        kwargs['model_reward'] = r.item()
        return state, action, reward, done, info, kwargs


class EnvRunner:
    """
    environment loop with pluggable observers

    to attach an observer implement EnvObserver interface and use attach()

    filters to process the steps are supported, and data enrichment is possible
    by adding to the kwargs dict
    """
    def __init__(self, args, env, action_pipeline, seed=None):
        self.args = args
        self.env = env
        if seed is not None:
            env.seed(seed)
        self.observers = OrderedDict()
        self.step_filters = OrderedDict()
        self.action_pipeline = action_pipeline

    def attach_observer(self, name, observer):
        self.observers[name] = observer

    def detach_observer(self, name):
        del self.observers[name]

    def append_step_filter(self, name, filter):
        self.step_filters[name] = filter

    def observer_reset(self):
        for name, observer in self.observers.items():
            observer.reset()

    def observe_step(self, state, action, reward, done, info, **kwargs):
        for name, filter in self.step_filters.items():
            state, action, reward, done, info, kwargs = filter(state, action, reward, done, info, **kwargs)
        for name, observer in self.observers.items():
            observer.step(state, action, reward, done, info, **kwargs)

    def observer_episode_end(self):
        for name, observer in self.observers.items():
            observer.done()

    def render(self, render, delay):
        if render:
            self.env.render()
            time.sleep(delay)

    def episode(self, args, policy, render=False, delay=0.01):
        with torch.no_grad():
            self.observer_reset()
            state, reward, done, info = self.env.reset(), 0.0, False, {}
            action_dist, sampled_action, action = self.action_pipeline(args, state, policy)
            self.observe_step(state, action, reward, done, info, action_dist=action_dist, sampled_action=sampled_action)
            self.render(render, delay)
            while not done:
                state, reward, done, info = self.env.step(action)
                action_dist, sampled_action, action = self.action_pipeline(args, state, policy)
                self.observe_step(state, action, reward, done, info, action_dist=action_dist, sampled_action=sampled_action)
                self.render(render, delay)

            self.observer_episode_end()