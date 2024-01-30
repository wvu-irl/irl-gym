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

# Let's assume I should normalize assumptions to be between 0 and 1.
# Then based on simulation accuracy, I can learn confidence weights for aligning parameters?
# remember belief will be computed as logsig (to norm to 1) of (weighted) distance from alignment

# need to compute confidence in assumption (can use KWIK stuff/CLT?)

#alignment can be done in two ways:
    # for each assumption, compute alignment for each model and weight by confidence in assumtion.
    # then based on their distances, get a belief for each model
    # multiply by "importance" of that assumption (how much it affects error)
    # perform dempster shafer combination to get belief in each model
# or
    # compute each alignment for each model
    #based on confidence in each assumption, compute weighted alignment
    # get distance from alignment then use that to estimate probability distribution?

#how to get weights since I don't know ground truth? -> use error in state transitions
    #based on assumption value and its correlation with state transition error, I can learn weights

class ForagingBattery(Assumption):
    """
    Assumption controls the battery level of the agent.
    
    # has base decay for time, movement decay, grab decay, drop decay, in_sun
    # let's just assume battery has fixed capacity of 100 for now.
    # also assume that in sunlight, battery recahrges with fixed rate of 0.1*dt
    
    #in the case of other decay rates, they can be fixed by enforcing limits on decay
    
    **Input**
    
    :param initial_state: (dict("battery_level", "decay")) value of the assumption, defaults {"battery_level": 100, "decay": 0}
    :param limits: (dict("battery_level", "decay")) limits of the assumption, defaults {"battery_level": [0, 100], "decay": [0, 1]}
        
    """
    def __init__(self, initial_state : dict = None, limits : dict = None):
            
        if limits is None:
            limits = {"battery_level": [0, 100], "decay": [0, 1]}
        elif "battery_level" not in limits or limits["battery_level"] is None:
            limits["battery_level"] = [0, 100]
        elif "decay" not in limits or limits["decay"] is None:
            limits["decay"] = [-100, 100]
                    
        # base decay, movement decay, grab decay, drop decay
        if initial_state is None:
            initial_state = {"battery_level": limits["battery_level"][1], "decay": limits["decay"][0]*np.ones(5)}
            initial_state["decay"][4] = 0.1
        super(ForagingBattery,self).__init__("Battery", initial_state, limits)
        
    def compute(self, s1, a = None, s2=None):
        # done over a single timestep.
        
        if "battery_level" in s1 and s1["battery_level"] is not None:
            
            dt = s2["time"] - s1["time"]
            s2["battery_level"] = s1["battery_level"] - self.__state["decay"][0]*dt
            
            if a == 1: # move
                x = s2["x"] - s1["x"]
                y = s2["y"] - s1["y"]
                d = np.sqrt(x**2 + y**2)
                s2["battery_level"] -= self.__state["decay"][1]*d
            elif a == 2: # grab
                s2["battery_level"] -= self.__state["decay"][2]
            elif a == 3: # drop
                s2["battery_level"] -= self.__state["decay"][3]
            
            if s2["battery_level"] < self.limits["battery_level"][0]:
                s2 = deepcopy(s1)
                s1["battery_level"] = 0
            if s2["is_home"]:
                s2["battery_level"] = self.limits["battery_level"][1]
        
            if "in_sun" in s1 and s1["in_sun"] is not None:
                if s1["in_sun"]:
                    s2["battery_level"] -= self.__state["decay"][4]/2
            if "in_sun" in s2 and s2["in_sun"] is not None:
                if s2["in_sun"]:
                    s2["battery_level"] -= self.__state["decay"][4]/2
            s2["battery_level"] = np.clip(s2["battery_level"], self.limits["battery_level"][0], self.limits["battery_level"][1])
        
        return s2  
    
    def update(self, s1 = None, a = None, s2 = None):
        bat_diff = s2["battery_level"] - s1["battery_level"]
        dt = s2["time"] - s1["time"]

        if a == 0 or self.__limits["decay"][a][1] == self.__limits["decay"][a][0]:
            if self.__limits["decay"][a][1] == self.__limits["decay"][a][0]:
                bat_diff -= self.__state["decay"][a]
            self.__state["decay"][0] = self.__alpha*self.__state["decay"][0] + (1-self.__alpha)*bat_diff/dt
        else:
            total = self.__state["decay"][0] + self.__state["decay"][a]
            if total == 0:
                share = [0.5, 0.5]
            else:
                share = [self.__state["decay"][0]/total, self.__state["decay"][a]/total]

            self.__state["decay"][0] = self.__alpha*self.__state["decay"][0] + (1-self.__alpha)*(bat_diff/dt)*share[0]
            
            if a == 1: 
                x = s2["x"] - s1["x"]
                y = s2["y"] - s1["y"]
                d = np.sqrt(x**2 + y**2)
                
                self.__state["decay"][1] = self.__alpha*self.__state["decay"][1] + (1-self.__alpha)*(bat_diff/d)*share[1]
            else:
                self.__state["decay"][a] = self.__alpha*self.__state["decay"][a] + (1-self.__alpha)*(bat_diff/dt)*share[1]
        for i in range(len(self.__state["decay"])):
            self.__state["decay"][i] = np.clip(self.__state["decay"][i], self.__limits["decay"][i][0], self.limits["decay"][i][1])
            
    def compute_alignment(self, params):
        """
        Computes the alignment of the assumption with the parameters
        
        :param params: (dict) parameters to compare to
        :return: (float) alignment
        """
        alignment = 0
        for i in range(len(self.__state["decay"])):
            alignment += (self.__state["decay"][i]-params["decay"][i])**2
        alignment = np.sqrt(alignment)
        return alignment

class ModelCost(Assumption):
    """
    Expense for running a model
    """
    def __init__(self, initial_state : dict = None, limits : dict = None):
            
        if limits is None:
            limits = {"cost": [0, 500]}
        elif "cost" not in limits or limits["cost"] is None:
            limits["cost"] = [0, 500]
                    
        # base decay, movement decay, grab decay, drop decay
        if initial_state is None:
            initial_state = {"cost": 0}
        super(ModelCost,self).__init__("Cost", initial_state, limits)
        
    def update(self, s1, a, s2):
        raise ValueError("ModelCost does not support update")
    
    def compute(self, s1, a = None, s2 = None):
        dt = s2["time"] - s1["time"]
        self.__state["cost"] = self.__alpha*self.__state["cost"] + (1-self.__alpha)*dt
    
    def alignment(self, params):
        """
        Computes the alignment of the assumption with the parameters
        
        :param params: (dict) parameters to compare to
        :return: (float) alignment
        """
        return abs(self.__state["cost"]-params["cost"])

#None assumption which always returns true and spits value back out

# Continuity
# Markov Property

#heading
#assumption controller (should go into continuity?)
#targets
#obstacles
#grab/drop probability
#slip (linear, angular) How to deal with grid? Make random? #perhaps make this random output of  motor?
#computational expense
#reward

#termination criterion is assumption for simulator. 
 # in continuous case it could be sim_time, time_step, or terminal __state (or just reaching coordinate)
 # in discrete case, it could be as well. 
 
# for termination criterion (dt, T, is_terminal)
    # while is task not completed or max_time not exceed, 
        # simulate timestep
# so let dt, T, and is_terminal (radius, dropped, picked_up) be assumptions
    # picked_up has its own timesteps too just assume 1 for now

#assume access to a responsible controller. 

# need to be able to choose fixed or variable timestep


#assume then that in the discrete case, agent can only transition to neighboring cells in a single timestep.
# so agent only needs to be concerned about neighboring cells and if they contain obstacles. 
# otherwise assume that the environment is consistent so that all cells behave the same unless otherwise specified



#discetization should be commensurate. So the agent has to end up in neighboring cell.
#so when doing mapping between __states, need to make sure space and velocity are equivalently mapped...
# so maybe use a nominal timestep based on drive velocity and discretization of space. 
# -> so define spatial discretization as a function of velocity and nomimal timestep for how far agent "could" get
# assume slip is necessarily subtractive

#maybe due partial observability too

#what is a typical update step going to look like?
#assume continuous for now. 
#assume action is to get to a waypoint (may be spatiotemporal) that way grid and control/graph can be handled same.
#while __state not at waypoint, or __state not terminal, update __state
    # attempt action and update __states
    #first is continuity which handles motion
    # base slip on motion
    
    #attempt action grab and update __states
    
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