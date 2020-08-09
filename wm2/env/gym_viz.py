from collections import deque
from matplotlib import pyplot as plt
import gym
import numpy as np


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


class VizPanel:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(24, 16), dpi=80, facecolor='w', edgecolor='k', )
        self.fig.canvas.set_window_title('Lander Viz')
        self.current_panel = 1
        self.panels = (5, 3)
        self.panels_array = {}

    def add_panel(self, label):
        self.panels_array.update({label: LiveLine(self.fig, self.panels, self._next_panel, label=label)})

    @property
    def _next_panel(self):
        current_panel = self.current_panel
        self.current_panel += 1
        return current_panel

    def reset(self):
        for key in self.panels_array:
            self.panels_array[key].reset()

    def update(self, info):
        for key in info:
            self.panels_array[key].dq.append(info[key])

    def draw(self):
        for key in self.panels_array:
            self.panels_array[key].draw()
        self.fig.canvas.draw()


def make_np_index(map):
    name = []
    ind = []
    for key, index in map.items():
        name.append(key)
        ind += [index]
    return name, np.stack(ind)


def get_size(space):
    if issubclass(type(space), gym.spaces.Discrete):
        return 1
    if issubclass(type(space), gym.spaces.Box):
        if len(space.shape) == 1:
            return space.shape[0]
        else:
            raise Exception('VizWrapper only supports 1D state spaces')


class VizWrapper(gym.Wrapper):
    """  will draw trajectories for states and actions
    to configure pass a dict that maps index to name

    state_map =  {
        'x_pos': 0,
        'y_pos': 1,
        'x_vel' : 2,
        'y_vel' : 3,
    }

    else it will try to autodiscover the state space from the env definition

    """

    def __init__(self, env, state_map=None, action_map=None):
        super().__init__(env)
        self.viz = VizPanel()
        self.configured = False
        self.state_map = {} if state_map is None else state_map
        self.state_index = None
        self.action_map = {} if action_map is None else action_map
        self.action_index = {}

        if state_map is None:

            if not hasattr(self.env, 'observation_space'):
                raise Exception('env.observation_space not defined, define or pass a config')

            for i in range(get_size(self.env.observation_space)):
                name = f'state {i}'
                self.viz.add_panel(name)
                self.state_map[name] = i
        else:
            for key in state_map:
                self.viz.add_panel(key)

        if action_map is None:

            if not hasattr(self.env, 'action_space'):
                raise Exception('env.action_space not defined, define it or pass a config')

            for i in range(get_size(self.env.action_space)):
                name = f'action {i}'
                self.viz.add_panel(name)
                self.action_map[name] = i
        else:
            for key in action_map:
                self.viz.add_panel(key)

        self.state_map, self.state_index = make_np_index(self.state_map)
        self.action_map, self.action_index = make_np_index(self.action_map)

    def update_state(self, state):
        subset = np.take(state, self.state_index)
        update = {}
        for key, s in zip(self.state_map, subset):
            update[key] = s
        self.viz.update(update)

    def update_action(self, state):
        subset = np.take(state, self.action_index)
        update = {}
        for key, s in zip(self.action_map, subset):
            update[key] = s
        self.viz.update(update)

    def reset(self, **kwargs):
        self.viz.reset()
        state = self.env.reset(**kwargs)
        self.update_state(state)
        return state

    def step(self, action):
        self.update_action(action)
        state, reward, done, info = self.env.step(action)
        self.update_state(state)
        if done:
            self.viz.draw()
        return state, reward, done, info

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

