"""
This module contains the assumptions for the foraging environment
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Jared Beard"

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from copy import deepcopy
import logging

import numpy as np

from irl_gym.envs.structures.modular.modular import *

class Battery(Assumption):
    """
    Battery assumption controls the battery level of the agent.

    **Input**
    
    :param initial_state: (dict("battery_level","decay")) value of the assumption
    
    
    """
    def __init__(self, initial_state : dict = None, limits : dict = None):
        
        if limits is None:
            limits = {"battery_level": [0, 100], "decay": [0, 1]}
        elif "battery_level" not in limits or limits["battery_level"] is None:
            limits["battery_level"] = [0, 100]
        elif "decay" not in limits or limits["decay"] is None:
            limits["decay"] = [0, 1]
        #perhaps I need a way to add multiple decay params
        
        if initial_state is None:
            initial_state = {"battery_level": limits["battery_level"][1], "decay": limits["decay"][0]}
        
        super().__init__("Battery", initial_state, limits)
        
    def compute(self, s1, s2=None):
        
        #what if motion not straight line? I think I should assume this gets called between two discrete points
        # so if there is a controller it does every timestep, and if there is not, it does every action
        
        if "battery_level" in s1 and s1["battery_level"] is not None:
            x = s2["x"] - s1["x"]
            y = s2["y"] - s1["y"]
            d = np.sqrt(x**2 + y**2)
            
            s2["battery_level"] = s1["battery_level"] - s1["decay"]*d
            if s2["battery_level"] < super().limits["battery_level"][0]:
                s2 = deepcopy(s1)
                s1["battery_level"] = 0
            if s2["is_home"]:
                s2["battery_level"] = super().limits["battery_level"][1]
        
        is_terminal = False
        if s2["battery_level"] == 0:
            is_terminal = True
            
        return s2, is_terminal   
    
    def update(self, state):
        # let's do estimation proces externally for now
        
        if super().check_limits(state):
            super.__state = state   

#Continuity
# Markov Property

#heading
#assumption controller (should go into continuity?)
#targets
#obstacles
#grab/drop probability
#slip (linear, angular) How to deal with grid? Make random?
#computational expense
#reward

#maybe due partial observability too

#what is a typical update step going to look like?
#assume continuous for now. 
#assume action is to get to a waypoint (may be spatiotemporal) that way grid and control/graph can be handled same.
#while state not at waypoint, or state not terminal, update state
    # attempt action and update states
    #first is continuity which handles motion
    # base slip on motion
    
    #attempt action grab and update states
    
    # base battery on slip + motion + grab
    
    #compute reward and update
    
#so target map can be assumption supplied to decision maker. -> probably want to wrap this in another class, if so can also add partial observability
#similarly computational expense is applied to decision maker.


class Continuity(Assumption):
    """
    Continuity assumption controls the discretization of the environment.
    
    If value is retained Values will be accepted as integers or 

    **Input**
    
    :param value: (float or Numeric) value of the assumption
    
    """
    pass