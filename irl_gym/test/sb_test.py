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

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import gymnasium as gym

from collisions import *

from sb_base import SBBase
from sb_support import SBSupport
from spawn_flowers import *

params = json.load(open(current+'/sb_params.json'))

env = gym.make("stickbug/SBEnv-v0", max_episode_steps=params["max_episode_steps"], params=params)
env.reset()

# print(type(params["base"]))
base = SBBase(params["base"])
support = SBSupport(params["support"], params["observation"])
orchard = Orchard(params["orchard"], params["row"], params["plant"])

plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 8), facecolor='black')
# ax = fig.add_subplot(111, projection='3d', facecolor='black')
ax = Axes3D(fig, auto_add_to_figure=False)

fig.add_axes(ax)
# Set labels and title
lim = 5

###

#Need to double check that pasisng flowers up didn't break anything. Pulled deepcopy out of plant in fet flowers method
# add successfully counted counter to orchard, row, etc
# need to check pollination to see if it gets updated as pollinated in the orchard

###
fn = fig.number

cnt = 0
# def update(frame, fig, ax):
while plt.fignum_exists(fn):
    # print(frame)
    ax.clear()# =deepcopy(ax)# = deepcopy(ax1)
    # orchard.plot(fig,ax,plot=False)

    dt = 0.1
    # base.go_2_position([0,0,0,cnt*0.1])
    base.step({"mode":"velocity",
               "command":[0,-1,0.1],
             })
    # base.go_2_position([0,0,0,0])
    # cnt += 1
    support.update(base.get_abs_support())
    
    # act = {"TR": {"hand": {"position":[2,-1.25,1.75]}}, "TL": {"hand": {"position":[0,1.25,1.25]}}, "BR": {"joint": {"velocity":[0,0,0.1]}}, "BL": {"joint": {"velocity":[0,0,0.1]}}}
    # act = {"TR": {"hand": {"position":[0.75,-0.75,1.75]}}, "TL": {"hand": {"position":[0.75,0.75,1.25]}}, "BR": {"joint": {"velocity":[0,0,0.1]}}, "BL": {"joint": {"velocity":[0,0,0.1]}}}
    act = {"TR": {"mode":"velocity",
                  "command":[0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                  "is_joint": False,
                  "is_relative": False,
                  "pollinate":False}
          }#, "TL": {"hand": {"position":[0.75,0.75,1.25]}}, "BR": {"joint": {"velocity":[0,0,0.1]}}, "BL": {"joint": {"velocity":[0,0,0.1]}}}
    # act = {"TR": {"joint": {"position":[1.75,0,0]}}, "TL": {"joint": {"position":[1.5,0,np.pi/2]}}, "BR": {"joint": {"velocity":[0,0,0.1]}}, "BL": {"joint": {"velocity":[0,0,0.1]}}}
    # cam_act = {"MR":{"cam":{"velocity":[0,0.1]}} }
    # act = {**act,**cam_act}
    support.step(act,dt=dt)
    # hp = BoundHexPrism([0,2.5,0], 0.5, 0.5, 0)
    # hp.plot(fig,ax,plot=False)
    # print(hp.get_points())
    
    b = base.get_abs_state()
    dist = params["support"]["mem_length"]["bicep"] + params["support"]["mem_length"]["forearm"] + params["observation"]["camera"]["distance"]
    dist *= 3
    flowers = deepcopy(orchard.get_flowers(b["pose"][0:3], dist))
    obs_flowers = support.observe_flowers(flowers)
    # print("me")
    # for arm in obs_flowers:
    #     print("arm ")
    #     for i in range(len(obs_flowers[arm])):
    #         print(obs_flowers[arm][i]["position"])
            #ax.scatter(flower["position"][0],flower["position"][1],flower["position"][2],color='black')
        
    base.plot(fig,ax,plot=False)
    support.plot(fig,ax,plot=False)

    fig.set_facecolor('black')
    ax.set_xlabel('X-axis', color='white')
    ax.set_xlim([-lim,2*lim])
    ax.set_ylabel('Y-axis', color='white')
    ax.set_ylim([-lim,2*lim])
    ax.set_zlabel('Z-axis', color='white')
    ax.set_zlim([0,lim])
    ax.set_title('Stickbug Arm Behavior Simulator', color='white')
    
    # ax.elev = 0
    # ax.azim = 270
    # ax.elev = 0
    # ax.azim = 270
    # xz 0, 270
    # yz 0, 0
    # xy -90, 0
    # hex.update(heading = heading)
    # hex.plot(fig,ax,plot=False)
    
    plt.pause(0.01)
    # plt.draw()

# except:
#     curses.endwin()
#     raise

# sim_time = 1000
# ani = FuncAnimation(fig, update, frames=range(0, sim_time), fargs=(fig, ax),repeat=False,interval=1)
# ani.save('stickbug_arm_sim.mp4', writer='ffmpeg', fps=30)