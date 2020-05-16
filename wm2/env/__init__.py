from gym.envs.registration import registry, register, make, spec
from wm2.env.LunarLander_v3 import LunarLander
from wm2.env.LunarLander_v3 import LunarLanderContinuous
import pybulletgym

register(
    id='LunarLander-v3',
    entry_point='wm2.env:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuous-v3',
    entry_point='wm2.env:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='HalfCheetahPyBulletEnv-v1',
    entry_point='wm2.env.half_cheetah:HalfCheetahBulletEnv',
    max_episode_steps=1000,
    reward_threshold=3000.0
)
