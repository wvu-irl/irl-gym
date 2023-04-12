import irl_gym
import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt

#"log_level": "DEBUG",
param = {"render": "plot", "dimensions": [20,20], "cell_size": 20, "goal": [10,13], "save_frames":False}
env = gym.make("irl_gym/Sailing-v0", max_episode_steps=5, params=param)
env.reset()
done = False
while not done:
    s, r, done, is_trunc, _ = env.step(4)
    print(s, r)
    plt.pause(1)
    env.render()
    print()
    print(env.get_actions(s))
    
    done = done or is_trunc
