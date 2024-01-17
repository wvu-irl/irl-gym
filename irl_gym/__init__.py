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
    id='irl_gym/AirHockey-v0',
    entry_point='irl_gym.envs:AirHockeyEnv',
    max_episode_steps=200,
    reward_threshold = None,
    disable_env_checker=True,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "params":
        {
            "freq": 120,
            "screenSize":[1000,1000], 
            "hitterPose": [250,250],
            "hitterRadius":50,
            "hitterMass": 1,
            "numPucks": 20,
            "puckRadius": 40,
            "puckMass": 1,
            "obs_type":"pose",
            "goalPose": [0,100],
            "goalHigh": [100,800],
            "energyLoss":0.05,
            "friction": 0.95,
            "maxVel": 1500
        }
    }
)
# print("after register")
# [ print(env) for env in gym_envs.registry.keys() if "irl" in env ]
