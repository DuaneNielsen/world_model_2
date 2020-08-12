import gym
import pybullet_envs


def register(id, entry_point, force=True, **kwargs):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point,
        **kwargs
    )


register(
    id='LunarLanderContinuous-v3',
    entry_point='wm2.env.LunarLander_v3:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLander-v3',
    entry_point='wm2.env.LunarLander_v3:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='HalfCheetahPyBulletEnv-v1',
    entry_point='wm2.env.half_cheetah:HalfCheetahBulletEnv',
    max_episode_steps=1000,
    reward_threshold=3000.0
)

register(
    id='Orbiter-v1',
    entry_point='wm2.env.orbiter:Orbiter',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LinEnv-v1',
    entry_point='wm2.env.LinEnv:LinEnv',
    max_episode_steps=1000,
    reward_threshold=200,
)
