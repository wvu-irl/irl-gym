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

__all__ = ["StickbugPlanner", "RefereePlanner", "HungarianPlanner", "NaivePlanner", "RandomRefereePlanner"]

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
        self._timeouts = {}
        self._time = 0
        
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
        
        if self._timeouts == {}:
            for arm in state["arms"]:
                self._timeouts[arm] = 0
                
        # update global pollinated (in future will need to make version that accounts for stochasticity)
        for arm in state["flowers"]:
            if len(self._flowers) == 0:
                self._flowers = deepcopy(state["flowers"][arm])
            else:
                temp = deepcopy(self._flowers)
                for flower in state["flowers"][arm]:
                    is_found = False
                    for temp_flower in self._flowers:
                        if np.linalg.norm(np.array(flower["position"]) - np.array(temp_flower["position"])) < 1e-3:
                            is_found = True
                    if not is_found:
                        temp.append(flower)
                self._flowers = temp
        
        for arm in state["pollinated"]:
            if len(self._pollinated) == 0:
                if len(state["pollinated"][arm]) > 0:
                    self._pollinated.append(deepcopy(state["pollinated"][arm]))
            elif len(state["pollinated"][arm]) > 0:
                temp = deepcopy(self._pollinated)
                is_found = False
                # print(self._pollinated)
                for temp_flower in self._pollinated:
                    # print(state["pollinated"][arm])
                    if np.linalg.norm(np.array(state["pollinated"][arm]["position"]) - np.array(temp_flower["position"])) <= 1e-3:
                        is_found = True
                if not is_found:
                    temp.append(state["pollinated"][arm])
                self._pollinated = temp
        
        temp_flow = []
        for flower in self._flowers:
            temp_flow.append(flower["position"])
        # print("FLOWERS", temp_flow)
        
        temp_pol = []
        for flower in self._pollinated:
            temp_pol.append(flower["position"])
        # print("POLLINATED ALL", temp_pol)
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
            arms.append(state["arms"][arm]["position"][0:3])
            
        unpollinated = []
        unpol_orientation = []
        pol_found = []
        for flower in self._flowers:
            is_found = False
            for pol in self._pollinated:
                if np.linalg.norm(np.array(flower["position"]) - np.array(pol["position"])) <= self._arm_params["pollination_radius"]*5/3.5:
                    is_found = True
            if not is_found:
                unpollinated.append(flower["position"])
                unpol_orientation.append(flower["orientation"])
            else:
                pol_found.append(flower["position"])
        
        # print("UNPOLLINATED", len(unpollinated), unpollinated)
        # print("POLLINATED F", len(pol_found), pol_found)
        
        arm_keys = list(self._targets.keys())
        for arm in arm_keys:
            if np.linalg.norm(np.array(state["arms"][arm]["position"][0:3]) - np.array(self._targets[arm])) <= self._arm_params["pollination_radius"]:
                del self._targets[arm]
                
        replan = False
        for arm in names:
            if arm not in self._targets:
                replan = True
                break
            
        hung_sln = hungarian_assignment(arms, unpollinated)
        # print("-----------------",hung_sln, "-----------------")
        
        action = {"arms":{}}
        hung_names = []
        for el in hung_sln:
            arm_name = names[el[0]]
            arm = self._arm_planner.get_arm(arm_name)
            hung_names.append(arm_name)
            super(type(arm), arm).evaluate(state)
            if arm.constraints_satisfied(unpollinated[el[1]],state) and self._timeouts[arm_name] <= self._arm_params["l3_timeout"]:
                pol = False
                if np.linalg.norm(np.array(unpollinated[el[1]]) - np.array(state["arms"][arm_name]["position"][0:3])) <= self._arm_params["pollination_radius"]:
                    action["arms"][arm_name] = {"mode": "velocity", "is_joint": False, "command": [0,0,0,0,0,0,0], "pollinate": True, "is_relative": False}
                    # print("whyyy", unpollinated[el[1]], state["arms"][arm_name]["position"][0:3], np.linalg.norm(np.array(unpollinated[el[1]]) - np.array(state["arms"][arm_name]["position"][0:3])) )
                    ### Can't figure out why this is not working. In short I think that because flowers are close it gets back a different
                    ### flower than the one it is close to. But this shouldn't matter because sb_arm only returns true for pollinated when
                    ### the arm tries to pollinate the first time. So I would think it gets the right one the next time...
                    #so here;s my dumb fix.
                    self._pollinated.append({"position":unpollinated[el[1]], "orientation":unpol_orientation[el[1]]})
                else:
                    action["arms"][arm_name] = {"mode": "position", "is_joint": False, "command": list(unpollinated[el[1]]) + [0,0,0,0], "pollinate": False, "is_relative": False}
                if arm_name not in self._targets or np.linalg.norm(np.array(self._targets[arm_name]) - np.array(unpollinated[el[1]])) > 1e-3:
                    self._timeouts[arm_name] = 0
                else:
                    self._timeouts[arm_name] += state["time"] - self._time
                self._targets[arm_name] = unpollinated[el[1]]
            else:
                joints = deepcopy(self._arm_params["joint_rest"])
                if "R" in arm_name:
                    joints[0] *= -1
                    joints[1] *= -1
                action["arms"][arm_name] = {"mode": "position", "is_joint": True, "command": [0,0,state["arms"][arm_name]["position"][2]] + joints + [0,0], "pollinate": False, "is_relative": False}
                if self._timeouts[arm_name] <= self._arm_params["l3_timeout"]+self._arm_params["l1_timeout"]:
                    self._timeouts[arm_name] = 0
                
        remaining = list(set(names) - set(hung_names))
        for arm in remaining:
            joints = deepcopy(self._arm_params["joint_rest"])
            if "R" in arm:
                joints[0] *= -1
                joints[1] *= -1
            action["arms"][arm] = {"mode": "position", "is_joint": True, "command": [0,0,state["arms"][arm]["position"][2]] + joints + [0,0], "pollinate": False, "is_relative": False}
            self._timeouts[arm] = 0
        
        self._time = state["time"]
        
        return action
          
class RandomRefereePlanner(StickbugPlanner):
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
        super(RandomRefereePlanner, self).__init__(params)
        
        self._log.debug("Init Referee Planner")
        
        self._flowers = []
        self._pollinated = []
        self._conflict = {}
        self._conflict_timers = {}
        self._num_conflicts = {"interaction":0, "flower_assignment":0, "no_flowers":0, "total":0}
        
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
                
        for arm in state["flowers"]:
            if len(self._flowers) == 0:
                self._flowers = deepcopy(state["flowers"][arm])
            else:
                temp = deepcopy(self._flowers)
                for flower in state["flowers"][arm]:
                    is_found = False
                    for temp_flower in self._flowers:
                        if np.linalg.norm(np.array(flower["position"]) - np.array(temp_flower["position"])) < 1e-3:
                            is_found = True
                    if not is_found:
                        temp.append(flower)
                self._flowers = temp
        
        for arm in state["pollinated"]:
            if len(self._pollinated) == 0:
                if len(state["pollinated"][arm]) > 0:
                    self._pollinated.append(deepcopy(state["pollinated"][arm]))
            elif len(state["pollinated"][arm]) > 0:
                temp = deepcopy(self._pollinated)
                is_found = False
                # print(self._pollinated)
                for temp_flower in self._pollinated:
                    # print(state["pollinated"][arm])
                    if np.linalg.norm(np.array(state["pollinated"][arm]["position"]) - np.array(temp_flower["position"])) <= 1e-3:
                        is_found = True
                if not is_found:
                    temp.append(state["pollinated"][arm])
                self._pollinated = temp
        
        temp_flow = []
        for flower in self._flowers:
            temp_flow.append(flower["position"])
        # print("FLOWERS", temp_flow)
        
        temp_pol = []
        for flower in self._pollinated:
            temp_pol.append(flower["position"])
        # print("POLLINATED ALL", temp_pol)
        
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
            arms.append(state["arms"][arm]["position"][0:3])
            
        unpollinated = []
        unpol_orientation = []
        pol_found = []
        for flower in self._flowers:
            is_found = False
            for pol in self._pollinated:
                if np.linalg.norm(np.array(flower["position"]) - np.array(pol["position"])) <= self._arm_params["pollination_radius"]*5/3.5:
                    is_found = True
            if not is_found:
                unpollinated.append(flower["position"])
                unpol_orientation.append(flower["orientation"])
            else:
                pol_found.append(flower["position"])
        
        # print("UNPOLLINATED", len(unpollinated), unpollinated)
        # print("POLLINATED F", len(pol_found), pol_found)
        
        # arm_keys = list(self._targets.keys())
        # for arm in arm_keys:
        #     if np.linalg.norm(np.array(state["arms"][arm]["position"][0:3]) - np.array(self._targets[arm])) <= self._arm_params["pollination_radius"]:
        #         del self._targets[arm]
        
        
        action = {"arms":{}}
        # action["base"] = self._base_planner.evaluate(state["base"]) # base
        action["arms"] = self._arm_planner.evaluate(state) # arm
        
        prev_conflicts = deepcopy(self._conflict)
        nonconflict = list(set(names)-set(self._conflict.keys()))
        for arm in nonconflict:
            temp_arm = self._arm_planner.get_arm(arm)
            flower_check = not len(temp_arm.reachable) and temp_arm.get_state() != "3"

            conflict_check = True
            for arm2 in state["arms"]:
                if arm != arm2:
                    for bd in state["arms"][arm]["bounds"]:
                        bound = state["arms"][arm]["bounds"][bd]
                        for bd2 in state["arms"][arm2]["bounds"]:
                            if bound.collision(state["arms"][arm2]["bounds"][bd2]):
                                conflict_check = False
                                joints = deepcopy(self._arm_params["joint_rest"])
                                if "R" in arm:
                                    joints[0] *= -1
                                    joints[1] *= -1
                                z = deepcopy(state["arms"][arm]["position"][2])
                                if state["arms"][arm]["position"][2] > state["arms"][arm2]["position"][2]:
                                    z += 0.3
                                else:
                                    z -= 0.3
                                action["arms"][arm] = {"mode": "position", "is_joint": True, "command": [0,0,z] + joints + [0,0], "pollinate": False, "is_relative": False}
                                self._conflict_timers[arm] = 0
                                self._conflict[arm] = action["arms"][arm]
                                self._num_conflicts["interaction"] += 1
                                                          
            if flower_check and conflict_check:
                in_hung = False
                if len(unpollinated):
                    hung_sln = hungarian_assignment(arms, unpollinated)
                    for el in hung_sln:
                        if arms[el[0]] == arm:
                            target = unpollinated[el[1]]
                            action["arms"][arm] = {"mode": "position", "is_joint": False, "command": list(target) + [0,0,0,0], "pollinate": False, "is_relative": False}
                            temp_arm.set_target(target)
                            self._conflict_timers[arm] = 0
                            self._conflict[arm] = action["arms"][arm]
                            in_hung = True
                            self._num_conflicts["flower_assignment"] += 1
                if not len(unpollinated) or not in_hung:
                    points = []
                    for arm2 in state["arms"]:
                        if ("L" in arm and "L" in arm2) or ("R" in arm and "R" in arm2):
                            points.append(state["arms"][arm2]["position"][2])
                    z_min = self._arbitration_params["base_pose"][2]+ self._arm_params["base_height"] + self._arm_params["support_offset"][2] + self._arm_params["buffer"]
                    z_max = z_min + self._arm_params["support_height"] - 2*self._arm_params["buffer"]
                    points.append(z_min)
                    points.append(z_max)
                    z = np.clip(state["arms"][arm]["position"][2], z_min+0.001, z_max-0.001)
                    z_min = np.max([el for el in points if el <= z])
                    z_max = np.min([el for el in points if el >= z])
                    z = np.random.uniform(z_min, z_max)
                    th1 = np.random.uniform(self._arm_params["joint_constraints"]["th1"]["min"], 0)
                    th2 = np.random.uniform(0, self._arm_params["joint_constraints"]["th2"]["max"])
                    if "R" in arm:
                        th1 *= -1
                        th2 *= -1
                    action["arms"][arm] = {"mode": "position", "is_joint": True, "command": [0,0,z,th1,th2,0,0], "pollinate": False, "is_relative": False}
                    self._conflict_timers[arm] = 0
                    self._conflict[arm] = action["arms"][arm]
                    self._num_conflicts["no_flowers"] += 1
            # elif flower_check and conflict_check:
            #     joints = deepcopy(self._arm_params["joint_rest"])
            #     if "R" in arm:
            #         joints[0] *= -1
            #         joints[1] *= -1
            #     action["arms"][arm] = {"mode": "position", "is_joint": True, "command": [0,0,state["arms"][arm]["position"][2]] + joints + [0,0], "pollinate": False, "is_relative": False}
            #     temp_arm.set_state("1")
            #     self._conflict_timers[arm] = 0
            #     self._conflict[arm] = action["arms"][arm]
                
        arm_keys = list(self._conflict.keys())
        for arm in arm_keys:
            if self._conflict_timers[arm] > self._arbitration_params["conflict_timer"]:
                self._conflict_timers[arm] = 0
                del self._conflict[arm]
            else:
                action["arms"][arm] = self._conflict[arm]
                self._conflict_timers[arm] += 1
        
        for arm in self._conflict:
            if arm not in prev_conflicts:
                self._num_conflicts["total"] += 1
            elif np.linalg.norm(np.array(prev_conflicts[arm]["command"])- np.array(self._conflict[arm]["command"])) > 1e-3:
                self._num_conflicts["total"] += 1
        
        return action 
    
    def get_num_conflicts(self):
        return self._num_conflicts    
    
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
        
        self._observation_points = []
        dx = self._arbitration_params["reachable_area"]["max"][0] - self._arbitration_params["reachable_area"]["min"][0]
        for i in np.linspace(self._arbitration_params["reachable_area"]["min"][0], self._arbitration_params["reachable_area"]["max"][0], int(np.ceil(dx/0.2))):
            dy = self._arbitration_params["reachable_area"]["max"][1] - self._arbitration_params["reachable_area"]["min"][1]
            for j in np.linspace(self._arbitration_params["reachable_area"]["min"][1], self._arbitration_params["reachable_area"]["max"][1], int(np.ceil(dy/0.2))):
                dz = self._arbitration_params["reachable_area"]["max"][2] - self._arbitration_params["reachable_area"]["min"][2]
                for k in np.linspace(self._arbitration_params["reachable_area"]["min"][2], self._arbitration_params["reachable_area"]["max"][2], int(np.ceil(dz/0.2))):
                    noise = np.random.normal(0,0.05,3)
                    x = i + noise[0]
                    y = j + noise[1]
                    z = k + noise[2]
                    x_check = x > self._arbitration_params["base_pose"][0] +self._arm_params["support_offset"][0] + 2*self._arm_params["buffer"]
                    z_min = self._arbitration_params["base_pose"][2]+ self._arm_params["base_height"] + self._arm_params["support_offset"][2] + self._arm_params["buffer"]
                    z_max = z_min + self._arm_params["support_height"] - 2*self._arm_params["buffer"]
                    z_check = z > z_min and z < z_max
                    left_y = self._arbitration_params["base_pose"][1] + self._arm_params["support_offset"][1]
                    right_y = self._arbitration_params["base_pose"][1] - self._arm_params["support_offset"][1]
                    x_support = self._arbitration_params["base_pose"][0] + self._arm_params["support_offset"][0]
                    arm_l = self._arm_params["mem_length"]["bicep"] + self._arm_params["mem_length"]["forearm"]
                    radius_check = np.sqrt((x-x_support)**2 + (y-left_y)**2) < arm_l or np.sqrt((x-x_support)**2 + (y-right_y)**2) < arm_l
                    if x_check and z_check and radius_check: 
                        self._observation_points.append([x,y,z])
        
        
        self._conflict_timers = {}
        
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
        
        #remove observation_points
        for arm in state["arms"]:
            l = len(self._observation_points)
            for i in range(l):
                point = deepcopy(self._observation_points[l-i-1])
                if np.linalg.norm(np.array(point) - np.array(state["arms"][arm]["position"][0:3])) < 0.1:
                    self._observation_points.pop(l-i-1)
                
        for arm in state["flowers"]:
            if len(self._flowers) == 0:
                self._flowers = deepcopy(state["flowers"][arm])
            else:
                temp = deepcopy(self._flowers)
                for flower in state["flowers"][arm]:
                    is_found = False
                    for temp_flower in self._flowers:
                        if np.linalg.norm(np.array(flower["position"]) - np.array(temp_flower["position"])) < 1e-3:
                            is_found = True
                    if not is_found:
                        temp.append(flower)
                self._flowers = temp
        
        for arm in state["pollinated"]:
            if len(self._pollinated) == 0:
                if len(state["pollinated"][arm]) > 0:
                    self._pollinated.append(deepcopy(state["pollinated"][arm]))
            elif len(state["pollinated"][arm]) > 0:
                temp = deepcopy(self._pollinated)
                is_found = False
                # print(self._pollinated)
                for temp_flower in self._pollinated:
                    # print(state["pollinated"][arm])
                    if np.linalg.norm(np.array(state["pollinated"][arm]["position"]) - np.array(temp_flower["position"])) <= 1e-3:
                        is_found = True
                if not is_found:
                    temp.append(state["pollinated"][arm])
                self._pollinated = temp
        
        temp_flow = []
        for flower in self._flowers:
            temp_flow.append(flower["position"])
        # print("FLOWERS", temp_flow)
        
        temp_pol = []
        for flower in self._pollinated:
            temp_pol.append(flower["position"])
        # print("POLLINATED ALL", temp_pol)
        
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
            arms.append(state["arms"][arm]["position"][0:3])
            
        unpollinated = []
        unpol_orientation = []
        pol_found = []
        for flower in self._flowers:
            is_found = False
            for pol in self._pollinated:
                if np.linalg.norm(np.array(flower["position"]) - np.array(pol["position"])) <= self._arm_params["pollination_radius"]*5/3.5:
                    is_found = True
            if not is_found:
                unpollinated.append(flower["position"])
                unpol_orientation.append(flower["orientation"])
            else:
                pol_found.append(flower["position"])
        
        # print("UNPOLLINATED", len(unpollinated), unpollinated)
        # print("POLLINATED F", len(pol_found), pol_found)
        
        # arm_keys = list(self._targets.keys())
        # for arm in arm_keys:
        #     if np.linalg.norm(np.array(state["arms"][arm]["position"][0:3]) - np.array(self._targets[arm])) <= self._arm_params["pollination_radius"]:
        #         del self._targets[arm]
        
        
        action = {"arms":{}}
        # action["base"] = self._base_planner.evaluate(state["base"]) # base
        action["arms"] = self._arm_planner.evaluate(state) # arm
        
        nonconflict = list(set(names)-set(self._conflict.keys()))
        for arm in nonconflict:
            temp_arm = self._arm_planner.get_arm(arm)
            flower_check = not len(temp_arm.reachable) and temp_arm.get_state() != "3"

            conflict_check = True
            for arm2 in state["arms"]:
                if arm != arm2:
                    for bd in state["arms"][arm]["bounds"]:
                        bound = state["arms"][arm]["bounds"][bd]
                        for bd2 in state["arms"][arm2]["bounds"]:
                            if bound.collision(state["arms"][arm2]["bounds"][bd2]):
                                conflict_check = False
                                joints = deepcopy(self._arm_params["joint_rest"])
                                if "R" in arm:
                                    joints[0] *= -1
                                    joints[1] *= -1
                                z = deepcopy(state["arms"][arm]["position"][2])
                                if state["arms"][arm]["position"][2] > state["arms"][arm2]["position"][2]:
                                    z += 0.3
                                else:
                                    z -= 0.3
                                action["arms"][arm] = {"mode": "position", "is_joint": True, "command": [0,0,z] + joints + [0,0], "pollinate": False, "is_relative": False}
                                self._conflict_timers[arm] = 0
                                self._conflict[arm] = action["arms"][arm]
                                
                    # if np.linalg.norm( np.array(state["arms"][arm]["position"][0:3]) - np.array(state["arms"][arm2]["position"][0:3]) ) < self._arbitration_params["conflict_threshold"]:
                    #     conflict_check = False
                    #     joints = deepcopy(self._arm_params["joint_rest"])
                    #     if "R" in arm:
                    #         joints[0] *= -1
                    #         joints[1] *= -1
                    #     action["arms"][arm] = {"mode": "position", "is_joint": True, "command": [0,0,state["arms"][arm]["position"][2]] + joints + [0,0], "pollinate": False, "is_relative": False}
                    #     temp_arm.set_state("1")
                    #     too_close = True
                    #     self._conflict_timers[arm] = 0
                    #     self._conflict[arm] = action["arms"][arm]
                    
                        
                    # if ("L" in arm and "L" in arm2) or ("R" in arm and "R" in arm2) or too_close:
                    #     if np.abs(state["arms"][arm]["position"][2] - state["arms"][arm2]["position"][2]) < 2*self._arm_params["buffer"]:
                    #         z_check = False
                    #         pos = deepcopy(state["arms"][arm]["position"])
                    #         if state["arms"][arm]["position"][2] > state["arms"][arm2]["position"][2]:
                    #             pos[2] += 0.3
                    #         else:
                    #             pos[2] -= 0.3
                            
                    #         if not too_close:
                    #             action["arms"][arm] = {"mode": "position", "is_joint": False, "command": pos, "pollinate": False, "is_relative": False}
                    #             temp_arm.set_target(state["arms"][arm]["position"][0:3])
                    #         else:
                    #             action["arms"][arm] = {"mode": "position", "is_joint": True, "command": [0,0,pos[2]] + joints + [0,0], "pollinate": False, "is_relative": False}
                    #             temp_arm.set_state("1")
                    #         self._conflict_timers[arm] = 0
                    #         self._conflict[arm] = action["arms"][arm]
                                           
            if flower_check and conflict_check and len(self._observation_points):
                if len(unpollinated):
                    nearest_flower = nearest_point(state["arms"][arm]["position"][0:3], unpollinated)
                reachable_obs = deepcopy(self._observation_points)
                l = len(reachable_obs)
                for i in range(l):
                    temp= deepcopy(self._observation_points[l-i-1])
                    if not temp_arm.constraints_satisfied(point, state):
                        reachable_obs.pop(l-i-1)
                if len(unpollinated) and temp_arm.constraints_satisfied(nearest_flower, state):
                    reachable_obs.append(nearest_flower)
                target = nearest_point(state["arms"][arm]["position"][0:3], self._observation_points)
                action["arms"][arm] = {"mode": "position", "is_joint": False, "command": list(target) + [0,0,0,0], "pollinate": False, "is_relative": False}
                temp_arm.set_target(target)
                self._conflict_timers[arm] = 0
                self._conflict[arm] = action["arms"][arm]
            elif flower_check and conflict_check:
                joints = deepcopy(self._arm_params["joint_rest"])
                if "R" in arm:
                    joints[0] *= -1
                    joints[1] *= -1
                action["arms"][arm] = {"mode": "position", "is_joint": True, "command": [0,0,state["arms"][arm]["position"][2]] + joints + [0,0], "pollinate": False, "is_relative": False}
                temp_arm.set_state("1")
                self._conflict_timers[arm] = 0
                self._conflict[arm] = action["arms"][arm]
                
        arm_keys = list(self._conflict.keys())
        for arm in arm_keys:
            if self._conflict_timers[arm] > self._arbitration_params["conflict_timer"]:
                self._conflict_timers[arm] = 0
                del self._conflict[arm]
            else:
                action["arms"][arm] = self._conflict[arm]
                self._conflict_timers[arm] += 1
        
        
        return action       
        
        # for arm in self._conflict:
        #     if self._conflict[arm] > self._arbitration_params["conflict_timer"]:
        #         del self._conflict[arm]
        #     else:
        #         del state["arms"][arm]
        #         self._conflict[arm] += 1
            # d = np.linalg( np.array(state["arms"][arm][0:3]) - np.array(self._conflict[arm]["command"][0:3]) )
            # if d < self._arbitration_params["conflict_threshold"]:
            #     del self._conflict[arm]
            # else:
            #     del state["arms"][arm]
        
        # action = {}
        # # action["base"] = self._base_planner.evaluate(state["base"]) # base                
        # action["arms"] = self._arm_planner.evaluate(state) # arm
        
        # for arm in self._conflict:
        #     action["arms"][arm] = self._conflict[arm]
        
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
            
    #     return action
    
    # def replan_arm(self, state, arm):
    #     return {arm:
    #                 {
    #                     "mode": "position", 
    #                     "is_joint": True,
    #                     "command": [0, 0, 0,self._arbitration_params["joint_rest"][0],self._arbitration_params["joint_rest"][0] , 0,0], 
    #                     "pollinate": True, 
    #                     "is_relative": True
    #                 }
    #             }
            
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
                # print(self._pollinated)
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
        # print("POLLINATED ALL", self._pollinated)
        # share pollinated flowers
        for arm in state["pollinated"]:
            state["pollinated"][arm] = deepcopy(self._pollinated)
               
        
        action = {}
        # action["base"] = self._base_planner.evaluate(state["base"]) # base                
        action["arms"] = self._arm_planner.evaluate(state) # arm
        
        return action
    
