"""
This module contains the Base Class for Stickbug planning
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
import functools 

from abc import ABC, abstractmethod

from irl_gym.utils.collisions import BoundCylinder

from irl_gym.test.other_stickbug.planners.planner import Planner
from irl_gym.test.other_stickbug.planners.arm_planners import MultiArmPlanner
from irl_gym.test.other_stickbug.planners.base_planners import BasePlanner
from irl_gym.test.other_stickbug.planners.plan_utils import *

__all__ = ["StickbugPlanner", "RefereePlanner", "HungarianPlanner", "NaivePlanner"]

class StickbugPlanner(Planner):
    """
    Base class for planning on Stickbug
    
    **Input**
    
    :param arbitration_params: (dict) dictionary of arbitration parameters
    :param base_params: (dict) dictionary of base parameters
    :param arm_params: (dict) dictionary of arm parameters
    
    :param log_level: (int) logging levels
    """
    def __init__(self, params = None):
        
        super(StickbugPlanner, self).__init__(params)
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        self._log.debug("Init Stickbug Base Planner")
        
        self._params = deepcopy(params)
        if "arm_params" in params and params["arm_params"] is not None:
            self._arm_params = params["arm_params"]
            self._arm_planner = MultiArmPlanner(params["arm_params"])
        if "base_params" in params and params["base_params"] is not None:
            self._base_params = params["base_params"]
            self._base_planner = BasePlanner(params["base_params"])
        self._arbitration_params = params["arbitration_params"]
    
    def reinit(self, state = None, action = None, s_prime = None):
        """
        Reinitialize Planners
        
        :param state: (dict) dictionary of state
        :param action: (dict) dictionary of action
        :param s_prime: (dict) dictionary of next state
        """
        raise NotImplementedError("reinit not implemented")
        
    def evaluate(self, state):
        """
        Get action from current state
        
        :param state: (dict) dictionary of state
        :return action: (dict) dictionary of action
        
        """
        action = {}
        if "base_params" in self._params and self._params["base_params"] is not None:
            action["base"] = self._base_planner.evaluate(state) # base                
        if "arm_params" in self._params and self._params["arm_params"] is not None:
            action["arms"] = self._arm_planner.evaluate(state)
        return action

class HungarianPlanner(StickbugPlanner):
    """
    Planning using the Hungarian Algorithm
    
    :param arbitration_params: (dict) dictionary of arbitration parameters
    :param base_params: (dict) dictionary of base parameters
    :param arm_params: (dict) dictionary of arm parameters
    
    :param log_level: (int) logging levels
    """
    def __init__(self, params=None):
        super(HungarianPlanner, self).__init__(params)
        
        self._log.debug("Init Referee Planner")
        
        self._flowers = []
        self._pollinated = []
        self._conflict = {}
        
        self._targets = {}
        
    def reinit(self, state=None, action=None, s_prime=None):
        """
        Reinitialize Planners
        
        :param state: (dict) dictionary of state
        :param action: (dict) dictionary of action
        :param s_prime: (dict) dictionary of next state
        """
        raise NotImplementedError("reinit not implemented")
        # reset the planners being refereed. Maybe reinitialize?
        
    def evaluate(self, state):
        """
        Planners for base and arms with referee interefering to limit actions
        
        :param state: (dict) dictionary of state
        """
        state = deepcopy(state)
                
        # update global pollinated (in future will need to make version that accounts for stochasticity)
        for arm in state["pollinated"]:
            if len(self._pollinated) == 0:
                if len(state["pollinated"][arm]) > 0:
                    self._pollinated.append(deepcopy(state["pollinated"][arm]))
            elif len(state["pollinated"][arm]) > 0:
                temp = deepcopy(self._pollinated)
                is_found = False
                print(self._pollinated)
                for temp_flower in self._pollinated:
                    # print(state["pollinated"][arm])
                    if np.linalg.norm(np.array(state["pollinated"][arm]["position"]) - np.array(temp_flower["position"])) <= self._arm_params["pollination_radius"]:
                        is_found = True
                if not is_found:
                    temp.append(state["pollinated"][arm])
                self._pollinated = temp
        for arm in state["flowers"]:
            if len(self._flowers) == 0:
                self._flowers = deepcopy(state["flowers"][arm])
            else:
                temp = deepcopy(self._flowers)
                for flower in state["flowers"][arm]:
                    is_found = False
                    for temp_flower in self._flowers:
                        if np.linalg.norm(np.array(flower["position"]) - np.array(temp_flower["position"])) < 5e-2:
                            is_found = True
                    if not is_found:
                        temp.append(flower)
                self._flowers = temp
        print("POLLINATED ALL", self._pollinated)
        # share pollinated flowers
        for arm in state["pollinated"]:
            state["pollinated"][arm] = deepcopy(self._pollinated)
        
        action = {}
        # action["base"] = self._base_planner.evaluate(state["base"]) # base                
        # action["arms"] = self._arm_planner.evaluate(state) # arm  
        names = []
        arms = []
        
        for arm in state["arms"]:
            names.append(arm)
            arms.append(state["arms"][arm]["position"])
            
        unpollinated = []
        for flower in self._flowers:
            is_found = False
            for pol in self._pollinated:
                if np.linalg.norm(np.array(flower["position"]) - np.array(pol["position"])) <= self._params["pollination_radius"]:
                    is_found = True
            if not is_found:
                unpollinated.append(flower["position"])
                
        for arm in self._targets:
            if np.linalg.norm(np.array(state["arms"][arm]["position"]) - np.array(self._targets[arm])) < self._arm_params["pollination_radius"]:
                del self._targets[arm]
                
        replan = False
        for arm in names:
            if arm not in self._targets:
                replan = True
                break
            
        hung_sln = hungarian_assignment(arms, unpollinated)
        
        action = {"arms":{}}
        for el in hung_sln:
            arm_name = names[el[0]]
            arm = self._arm_planner.get_arm(arm_name)
            
            if arm.constraints_satisfied(unpollinated[el[1]],state):
                action["arms"][arm_name] = {"mode": "position", "is_joint": False, "command": unpollinated[el[1]] + [0,0,0,0], "pollinate": True, "is_relative": False}
                self._targets[arm_name] = unpollinated[el[1]]
            else:
                action["arms"][arm_name] = {"mode": "position", "is_joint": False, "command": [0,0,state["arms"][arm_name]["position"][2]] + self._arm_params["joint_rest"] + [0,0], "pollinate": False, "is_relative": False}
        
        return action
          
    
    #will need to include memory for intereference actions
class RefereePlanner(StickbugPlanner):
    """
    Planning using the referee to handle conflicts
    
    :param arbitration_params: (dict) dictionary of arbitration parameters
    - :param base_referee: (str) type of arbitration for base
    - :param arm_referee: (str) type of arbitration for arm
    - :param reach: (float) distance to consider in reach
    :param base_params: (dict) dictionary of base parameters
    :param arm_params: (dict) dictionary of arm parameters
    
    :param log_level: (int) logging levels
    """
    def __init__(self, params=None):
        super(RefereePlanner, self).__init__(params)
        
        self._log.debug("Init Referee Planner")
        
        self._flowers = []
        self._pollinated = []
        self._conflict = {}
        
    def reinit(self, state=None, action=None, s_prime=None):
        """
        Reinitialize Planners
        
        :param state: (dict) dictionary of state
        :param action: (dict) dictionary of action
        :param s_prime: (dict) dictionary of next state
        """
        raise NotImplementedError("reinit not implemented")
        # reset the planners being refereed. Maybe reinitialize?
        
    def evaluate(self, state):
        """
        Planners for base and arms with referee interefering to limit actions
        
        :param state: (dict) dictionary of state
        """
        state = deepcopy(state)
                
        # update global pollinated (in future will need to make version that accounts for stochasticity)
        for arm in state["pollinated"]:
            if len(self._pollinated) == 0:
                if len(state["pollinated"][arm]) > 0:
                    self._pollinated.append(deepcopy(state["pollinated"][arm]))
            elif len(state["pollinated"][arm]) > 0:
                temp = deepcopy(self._pollinated)
                is_found = False
                print(self._pollinated)
                for temp_flower in self._pollinated:
                    # print(state["pollinated"][arm])
                    if np.linalg.norm(np.array(state["pollinated"][arm]["position"]) - np.array(temp_flower["position"])) <= self._arm_params["pollination_radius"]:
                        is_found = True
                if not is_found:
                    temp.append(state["pollinated"][arm])
                self._pollinated = temp
        for arm in state["flowers"]:
            if len(self._flowers) == 0:
                self._flowers = deepcopy(state["flowers"][arm])
            else:
                temp = deepcopy(self._flowers)
                for flower in state["flowers"][arm]:
                    is_found = False
                    for temp_flower in self._flowers:
                        if np.linalg.norm(np.array(flower["position"]) - np.array(temp_flower["position"])) < 5e-2:
                            is_found = True
                    if not is_found:
                        temp.append(flower)
                self._flowers = temp
        print("POLLINATED ALL", self._pollinated)
        # share pollinated flowers
        for arm in state["pollinated"]:
            state["pollinated"][arm] = deepcopy(self._pollinated)
               
        # filter out pollinated flowers
        # for arm in state["flowers"]:
        #     for flower in state["flowers"][arm]:
        #         for pollinated in self._pollinated:
        #             if np.sqrt(np.linalg.norm(np.array(flower["position"]) - np.array(pollinated["position"]))) < 4e-2:
        #                 state["flowers"][arm].remove(flower)
        
        for arm in self._conflict:
            if self._conflict[arm] > self._arbitration_params["conflict_timer"]:
                del self._conflict[arm]
            else:
                del state["arms"][arm]
                self._conflict[arm] += 1
            # d = np.linalg( np.array(state["arms"][arm][0:3]) - np.array(self._conflict[arm]["command"][0:3]) )
            # if d < self._arbitration_params["conflict_threshold"]:
            #     del self._conflict[arm]
            # else:
            #     del state["arms"][arm]
        
        action = {}
        # action["base"] = self._base_planner.evaluate(state["base"]) # base                
        action["arms"] = self._arm_planner.evaluate(state) # arm
        
        for arm in self._conflict:
            action["arms"][arm] = self._conflict[arm]
        
        # referee arbitration
        # if self._arbitration_params["base_referee"] == "density":
        #     workspace = BoundCylinder(state["base"]["position"], self._arbitration_params["reach"], self._params["base_params"]["workspace_height"])
        #     in_reach = []
        #     for flower in state["flowers"]:
        #         if workspace.contains(flower):
        #             in_reach.append(flower)
            
        #     if len(in_reach) > self._arbitration_params["density_threshold"]:
        #         action["base"] = {"mode": "velocity", "command": [0,0,0]}
        
        
        # if self._arbitration_params["arm_referee"] == "pick_and_place":
        #     for arm in state["arms"]:
        #         for arm2 in state["arms"]:
        #             if arm != arm2:
        #                 if np.linalg.norm( np.array(state["arms"][arm]["position"][0:3]) - np.array(state["arms"][arm2]["position"][0:3]) ) < self._arbitration_params["conflict_threshold"]:
        #                     d1 = np.linalg.norm( np.array(state["arms"][arm]["position"][0:3]) - np.array(action["arms"]["arm"]["command"][0:3]) )
        #                     d2 = np.linalg.norm( np.array(state["arms"][arm2]["position"][0:3]) - np.array(action["arms"]["arm2"]["command"][0:3]) )
                            
        #                     if d1 > self._arbitration_params["conflict_threshold"]:
        #                         action["arms"][arm]["command"][0:3] = self.replan_arm(state["arms"], arm)
        #                         self._conflict[arm] = action["arms"][arm]["command"][0:3]
        #                     if d2 > self._arbitration_params["conflict_threshold"]:
        #                         action["arms"][arm2]["command"][0:3] = self.replan_arm(state["arms"], arm2)
        #                         self._conflict[arm2] = action["arms"][arm2]["command"][0:3]
        # elif self._arbitration_params["arm_referee"] == "timed":
        #     for arm in state["arms"]:
        #         for arm2 in state["arms"]:
        #             if arm != arm2:
        #                 if np.linalg.norm( np.array(state["arms"][arm]["position"][0:3]) - np.array(state["arms"][arm2]["position"][0:3]) ) < self._arbitration_params["conflict_threshold"]:
        #                     self._conflict[arm] = 0
        #                     self._conflict[arm2] = 0
                            
        # for arm in self._conflict:
        #     action["arms"][arm] = self.replan_arm(state["arms"], arm)    
            
        return action
    
    def replan_arm(self, state, arm):
        return {arm:
                    {
                        "mode": "position", 
                        "is_joint": True,
                        "command": [0, 0, 0,self._arbitration_params["joint_rest"][0],self._arbitration_params["joint_rest"][0] , 0,0], 
                        "pollinate": True, 
                        "is_relative": True
                    }
                }
            
    # def replan_arm(self, state, arm):
        
    #     flowers = deepcopy(state["flowers"][arm])
    #     for flower in flowers:
    #         dmin = np.inf
    #         closest = arm
    #         for temp in state["arms"]:
    #             d = np.linalg( np.array(state["arms"][temp][0:3]) - np.array(flower) )
    #             if d < dmin:
    #                 closest = arm
    #                 dmin = d
                    
    #         if closest != arm:
    #             flowers.remove(flower)
        
    #     use_second = False
    #     if len(flowers) == 0:
    #         use_second = True
    #         flowers = deepcopy(state["flowers"][arm])
    #         # return {arm:{
    #         #             "mode": "position", 
    #         #             "is_joint": True,
    #         #             "command": [0, 0, 0,np.pi/4,np.pi/4, 0,0], 
    #         #             "pollinate": True, 
    #         #             "is_relative": True
    #         #             }
    #         #         }
    #     # else:
    #     nearest = flower 
    #     second_nearest = flower
    #     dmin = np.inf
    #     for flower in flowers:
    #         d = np.linalg( np.array(state["arms"][arm][0:3]) - np.array(flower) )
    #         if d < dmin:
    #             second_nearest = deepcopy(nearest)
    #             nearest = flower
    #             dmin = d
                
    #     if use_second:
    #         nearest = second_nearest
            
    #     return {arm:{
    #                 "mode": "position", 
    #                 "is_joint": False,
    #                 "command": nearest + [0,0] + state["arms"][arm][5:7],
    #                 "pollinate": True, 
    #                 "is_relative": False
    #                 }
    #             }
    
class NaivePlanner(StickbugPlanner):
    """
    Planning with no conflict handling
    
    :param arbitration_params: (dict) dictionary of arbitration parameters
    - :param base_referee: (str) type of arbitration for base
    - :param arm_referee: (str) type of arbitration for arm
    - :param reach: (float) distance to consider in reach
    :param base_params: (dict) dictionary of base parameters
    :param arm_params: (dict) dictionary of arm parameters
    
    :param log_level: (int) logging levels
    """
    def __init__(self, params=None):
        super(NaivePlanner, self).__init__(params)
        
        self._log.debug("Init Referee Planner")
        
        self._flowers = []
        self._pollinated = []
        self._conflict = {}
        
    def reinit(self, state=None, action=None, s_prime=None):
        """
        Reinitialize Planners
        
        :param state: (dict) dictionary of state
        :param action: (dict) dictionary of action
        :param s_prime: (dict) dictionary of next state
        """
        raise NotImplementedError("reinit not implemented")
        # reset the planners being refereed. Maybe reinitialize?
        
    def evaluate(self, state):
        """
        Planners for base and arms with referee interefering to limit actions
        
        :param state: (dict) dictionary of state
        """
        state = deepcopy(state)
                
        # update global pollinated (in future will need to make version that accounts for stochasticity)
        for arm in state["pollinated"]:
            if len(self._pollinated) == 0:
                if len(state["pollinated"][arm]) > 0:
                    self._pollinated.append(deepcopy(state["pollinated"][arm]))
            elif len(state["pollinated"][arm]) > 0:
                temp = deepcopy(self._pollinated)
                is_found = False
                print(self._pollinated)
                for temp_flower in self._pollinated:
                    # print(state["pollinated"][arm])
                    if np.linalg.norm(np.array(state["pollinated"][arm]["position"]) - np.array(temp_flower["position"])) <= self._arm_params["pollination_radius"]:
                        is_found = True
                if not is_found:
                    temp.append(state["pollinated"][arm])
                self._pollinated = temp
        for arm in state["flowers"]:
            if len(self._flowers) == 0:
                self._flowers = deepcopy(state["flowers"][arm])
            else:
                temp = deepcopy(self._flowers)
                for flower in state["flowers"][arm]:
                    is_found = False
                    for temp_flower in self._flowers:
                        if np.linalg.norm(np.array(flower["position"]) - np.array(temp_flower["position"])) < 5e-2:
                            is_found = True
                    if not is_found:
                        temp.append(flower)
                self._flowers = temp
        print("POLLINATED ALL", self._pollinated)
        # share pollinated flowers
        for arm in state["pollinated"]:
            state["pollinated"][arm] = deepcopy(self._pollinated)
               
        
        action = {}
        # action["base"] = self._base_planner.evaluate(state["base"]) # base                
        action["arms"] = self._arm_planner.evaluate(state) # arm
        
        return action
    
