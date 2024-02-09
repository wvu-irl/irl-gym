"""
This module contains tests for Stickbug Environment
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Trevor Smith"

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from copy import deepcopy
import logging

import json
import numpy as np
import math

import matplotlib.pyplot as plt

import gymnasium as gym

from irl_gym.utils.collisions import *

from irl_gym.support.stickbug.sb_base import SBBase
from irl_gym.support.stickbug.sb_support import SBSupport
from irl_gym.support.stickbug.spawn_flowers import *

params = json.load(open(current+'/sb_params.json'))
params = params["envs"][0]
print(params)
env = gym.make("irl_gym/StickbugEnv-v0", max_episode_steps=params["max_episode_steps"], params=params["params"])
env.reset()

# need to check pollination to see if it gets updated as pollinated in the orchard

# done = False
# while not done and plt.fignum_exists(env.get_fignum()):
#     a= {}
#     a["base"]={"mode":"velocity",
#                "command":[0,1,0.1],
#                 }
#     a["arms"] = {"TR": {"mode":"velocity",
#                   "command":[0.1,0.1,0.1,0.1,0.1,0.1,0.1],
#                   "is_joint": False,
#                   "is_relative": False,
#                   "pollinate":True}
#           }#, "TL": {"hand": {"position"
#     s, r, done, is_trunc, _ = env.step(a)
#     # print(s, r)
    
#     env.render()
#     plt.pause(0.5)

#     print(_)
#     # print(env.get_actions(s))
    
#     done = done or is_trunc

if params["params"]["save_gif"]:
    env.img_2_gif()
# sim_time = 1000
# ani = FuncAnimation(fig, update, frames=range(0, sim_time), fargs=(fig, ax),repeat=False,interval=1)
# ani.save('stickbug_arm_sim.mp4', writer='ffmpeg', fps=30)