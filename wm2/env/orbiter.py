from vpython import sphere, vector, rate, color, arrow, canvas, cross, triangle, vertex
from math import sqrt
import numpy as np
import gym
from gym.spaces import Box
from env.connector import EnvConnector
from torch import nn

G = 6.67e-11  # N kg^-2 m^2
au = 1.495978707e11
day = 60 * 60 * 24
year = day * 365.25

earth_mass = 5.972e24  # kg
sun_mass = 1.989e30  # kg
ship_mass = 400e3  # kg

""" normalization units """
square_au = au ** 2
max_thrust = 2000  # N


def area_tri(center, prev, curr):
    return 0.5 * cross((prev - center), (curr - center)).mag


def xyz(v):
    return [v.x/au, v.y/au, v.z/au]


class Reward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state):
        pass


class OrbiterConnector(EnvConnector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Orbiter(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
        self.action_space = Box(-1, +1, (2,), dtype=np.float32)

        self.t = 0
        self.dt = day / 2

        self.scene = canvas(width=1200, height=1200)
        self.ship = arrow(pos=vector(au, 0, 0), make_trail=False)
        self.sun = sphere(color=color.yellow, radius=0.1 * au)

        self.ship.r = vector(au, 0, 0)
        self.ship.m = ship_mass
        self.ship.pos = self.ship.r
        self.ship.prev = self.ship.r

        self.sun.r = vector(0, 0, 0)
        self.sun.m = sun_mass  # kg
        self.sun.pos = self.sun.r

        d = (self.ship.r - self.sun.r).mag
        self.ship.velocity = vector(0, sqrt(G * self.sun.m / d), 0)
        self.ship.axis = self.ship.velocity.norm() * 0.05 * au
        self.sun.velocity = vector(0, 0, 0)

    def reset(self):
        self.ship.r = vector(au, 0, 0)
        self.ship.m = 400e3
        self.ship.pos = self.ship.r
        self.ship.prev = self.ship.r
    
        self.sun.r = vector(0, 0, 0)
        self.sun.m = 1.989e30  # kg
        self.sun.pos = self.sun.r
    
        d = (self.ship.r - self.sun.r).mag
        self.ship.velocity = vector(0, sqrt(G * self.sun.m / d), 0)
        self.ship.axis=self.ship.velocity.norm() * 0.05 * au
        self.sun.velocity = vector(0, 0, 0)
    
        return np.array([*xyz(self.ship.pos), *xyz(self.ship.velocity), *xyz(self.sun.pos), *xyz(self.sun.velocity)])
    
    def step(self, action):
        f = vector(action[0], action[1], 0.0) * max_thrust
        r = self.ship.r - self.sun.r
        f += G * self.sun.m * self.ship.m * r / r.mag ** 3
        self.ship.velocity -= f / self.ship.m * self.dt
        self.ship.r, self.ship.prev = self.ship.r + self.ship.velocity * self.dt, self.ship.r
    
        # triangle(v0=vertex(pos=self.sun.pos, color=color.blue), v1=vertex(pos=self.ship.prev), v2=vertex(pos=self.ship.pos))
    
        reward = area_tri(self.sun.pos, self.ship.prev, self.ship.pos) / square_au * (year/self.dt)
    
        self.t += self.dt

        self.ship.pos, self.ship.axis = self.ship.r, self.ship.velocity.norm() * 0.05 * au

        return np.array([*xyz(self.ship.pos), *xyz(self.ship.velocity),
                         *xyz(self.sun.pos), *xyz(self.sun.velocity)]), \
               reward, self.t > 2 * year, {}

    def render(self, mode='human'):
        pass


if __name__ == '__main__':

    env = Orbiter()
    state, done = env.reset(), False
    while not done:
        f = np.array([0, 0])
        if env.t < 20 * day:
            r = vector(*state[0:3]) - vector(*state[6:9])
            f = cross(r.norm(), vector(0, 0, 1)) * 1500
            f = np.array([f.x, f.y])
        state, reward, done, info = env.step(f)
        print(state)
        rate(400)
