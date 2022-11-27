from gym.envs.registration import register


"""
Installs irl_gym envs

"""

register(
    id='irl_gym/GridWorld-v0',
    entry_point='irl_gym.envs:GridWorldEnv',
    max_episode_steps=100,
    reward_threshold = None,
    disable_env_checker=True,
    nondeterministic = True,
    order_enforce = True,
    autoreset = False,
    kwargs = 
    {
        "_params":
        {
            "agent": [20,20],
            "dimensions": [40,40],
            "goal": [10,10],
            "p": 0.1,
            "prefix": "/home/jared/ambiguity_ws/data/gridworld/",
            "reward_bounds": [0,1]
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
        "_params":
        {
            "agent": [25,5],
            "dimensions": [35,10],
            "goal": [10,10],
            "p": 0.1,
            "prefix": "/home/jared/ambiguity_ws/data/gridworld/",
            "reward_bounds": [0,1]
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
        "_params":
        {
            "agent": [20,20],
            "dimensions": [40,40],
            "goal": [10,10],
            "p": 0.1,
            "prefix": "/home/jared/ambiguity_ws/data/gridworld/",
            "reward_bounds": [0,1]
        }
    }
)