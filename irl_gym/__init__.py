from gymnasium.envs.registration import register
from gymnasium import envs as gym_envs

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))

"""
Installs irl_gym envs

"""
# print("before register")
# [ print(env) for env in gym_envs.registry.keys() if "irl" in env ]
register(
    id='irl_gym/GridWorld-v0',
    entry_point='irl_gym.envs:GridWorldEnv',
    max_episode_steps=100,
    reward_threshold = None,
    disable_env_checker=False,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "dimensions": [40,40],
            "goal": [10,10],
            "state": 
            {
                "pose": [20,20]
            },
            "r_radius": 5,
            "r_range": (-0.01, 1),
            "p": 0.1,
            "render": "none",
            "cell_size": 50,
            "prefix": current + "/plot/",
            "save_frames": False,
            "log_level": "WARNING"
        }
    }
)


register(
    id='irl_gym/GridTunnel-v0',
    entry_point='irl_gym.envs:GridTunnelEnv',
    max_episode_steps=100,
    reward_threshold = None,
    disable_env_checker=True,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "dimensions": [40,40],
            "goal": [10,10],
            "state": 
            {
                "pose": [20,20]
            },
            "r_radius": 5,
            "r_range": (-0.01, 1),
            "p": 0.1,
            "render": "none",
            "cell_size": 50,
            "prefix": current + "/plot/",
            "save_frames": False,
            "log_level": "WARNING"
        }
    }
)

register(
    id='irl_gym/Sailing-v0',
    entry_point='irl_gym.envs:SailingEnv',
    max_episode_steps=100 ,
    reward_threshold = None,
    disable_env_checker=True,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "dimensions": [40,40],
            "goal": [10,10],
            "state_offset": 15,
            "trap_offset": 17,
            "r_radius": 5,
            "r_range": (-400,1100),
            "p": 0.1,
            "render": "none",
            "cell_size": 50,
            "prefix": current + "/plot/",
            "save_frames": False,
            "log_level": "WARNING"
        }
    }
)

register(
    id='irl_gym/SailingBR-v0',
    entry_point='irl_gym.envs:SailingBREnv',
    max_episode_steps=100 ,
    reward_threshold = None,
    disable_env_checker=True,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "dimensions": [40,40],
            "goal": [10,10],
            "state_offset": 15,
            "trap_offset": 17,
            "r_radius": 5,
            "r_range": (-400,1100),
            "p": 0.1,
            "render": "none",
            "cell_size": 50,
            "prefix": current + "/plot/",
            "save_frames": False,
            "log_level": "WARNING"
        }
    }
)

register(
    id='irl_gym/StickbugEnv-v0',
    entry_point='irl_gym.envs:StickbugEnv',
    max_episode_steps=100,
    reward_threshold = None,
    disable_env_checker=False,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "r_range": [-0.1,1],
            "t_max": 100,
            "dt": 0.1,
            "prefix": current + "/plot/",
            "save_frames": False,
            "save_gif": False,
            "log_level": "WARNING"
        }
    }
)
# print("after register")
# [ print(env) for env in gym_envs.registry.keys() if "irl" in env ]