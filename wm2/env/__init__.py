from gym.envs.registration import registry, register, make, spec
from wm2.env.LunarLander_v3 import LunarLander
from wm2.env.LunarLander_v3 import LunarLanderContinuous


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
