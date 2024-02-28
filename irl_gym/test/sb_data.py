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

import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym

from irl_gym.utils.collisions import *

from irl_gym.support.stickbug.sb_base import SBBase
from irl_gym.support.stickbug.sb_support import SBSupport
from irl_gym.support.stickbug.spawn_flowers import *
from irl_gym.test.other_stickbug.planners.stickbug_planners import *

from irl_gym.test.file_utils import *

params = json.load(open(current+'/sb_params.json'))
params = params["envs"][0]
plan_params = json.load(open(current+'/other_stickbug/sb_plan_params.json'))

mode = "referee"

save_path = current + "/" + mode + str(params["params"]["support"]["num_arms"]) + "_data.csv"
print(save_path)

results = pd.DataFrame()

num_trials = 1000
i = 634
total = 0
while i < num_trials and total < 10000:
    print(i, mode, params["params"]["support"]["num_arms"])
    env = gym.make("irl_gym/StickbugEnv-v0", max_episode_steps=params["max_episode_steps"], params=params["params"])
    s, _ = env.reset()
    if mode == "naive":
        planner = NaivePlanner(plan_params["algs"][0]["params"])
    elif mode == "hungarian":
        planner = HungarianPlanner(plan_params["algs"][0]["params"])
    elif mode == "referee":
        planner = RandomRefereePlanner(plan_params["algs"][0]["params"])
    done = False
    
    try:
        result = {"alg": mode, "trial": i, "num_arms": params["params"]["support"]["num_arms"]}
        time = [0]*params["max_episode_steps"]
        num_pollinated = [0]*params["max_episode_steps"]
        j = 0
        while not done:# and plt.fignum_exists(env.get_fignum()):

            a = planner.evaluate(s)
            s, r, done, is_trunc, info = env.step(a)
            
            time[j] = info["time"]
            num_pollinated[j] = info["n_pollinated"]
            result["num_flowers"] = info["n_flowers"]
            
            j += 1
            done = done or is_trunc
        if j < params["max_episode_steps"]:
            for k in range(j, params["max_episode_steps"]):
                time[k] = time[j-1]
                num_pollinated[k] = num_pollinated[j-1]
        result["time"] = time
        result["num_pollinated"] = num_pollinated
        num_conflicts = planner.get_num_conflicts()
        result["total_conflicts"] = num_conflicts["total"]
        result["inter_arm_conflicts"] = num_conflicts["interaction"]
        result["flower_assignment_conflicts"] = num_conflicts["flower_assignment"]
        result["no_flowers_conflicts"] = num_conflicts["no_flowers"]

        results = pd.concat([results, pd.DataFrame(result)])
        data = import_file(save_path)
        data = pd.concat([data, pd.DataFrame(result)])
        export_file(data, save_path)
        i += 1
    except:
        print("error")
        total += 1