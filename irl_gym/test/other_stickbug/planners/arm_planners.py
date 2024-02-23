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

from abc import ABC, abstractmethod

from irl_gym.test.other_stickbug.planners.planner import Planner
from irl_gym.test.other_stickbug.planners.plan_utils import *
from irl_gym.utils.tf import *

class MultiArmPlanner(Planner):
    """
    Retains multiple arms for planning
    
    **Input**
    
    :param log_level: (int) logging level
    :param planner_type: (str) type of planner to use
    :param arms: (list) list of arms to plan for
    :param arm_params: (dict) dictionary of arm parameters
    """
    def __init__(self, params = None):
        
        super(MultiArmPlanner, self).__init__(params)
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        self._log.debug("Init Stickbug MultiArm Planner")
        
        self._arm_names = []
        if "num_arms" in params:
            naming = ["T", "M", "B", "BB", "BBB"]
        
            if params["num_arms"] % 2 != 0:
                l_arm = params["num_arms"] // 2
                r_arm = params["num_arms"] // 2 + 1
            else:
                l_arm = params["num_arms"] // 2
                r_arm = params["num_arms"] // 2
                
            for i in range(l_arm):
                self._arm_names.append(naming[i]+"L")
            for i in range(r_arm):
                self._arm_names.append(naming[i]+"R")
        
        self._arms = {}
        for arm in self._arm_names:
            self._arms[arm] = GreedyArmPlanner(arm, params)
            
        self._params = deepcopy(params)
    
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
        for arm in state["arms"]:
            action[arm] = self._arms[arm].evaluate(state)
        return action
    
    def get_arm(self, name):
        """
        Get arm from name
        
        :param name: (str) name of arm
        :return arm: (ArmPlanner) arm planner
        """
        return self._arms[name]
    
class ArmPlanner(Planner):
    """
    Base class for Stickbug planning
    
    **Input**
    
    :param mem_length: (dict) length of members {"bicep": <num>, "forearm": <num>}
    :param support_offset: (list) [x,y,z] offset of support
    :param joint_rest: (list) [bicep, forearm] rest position of joints
    :param timeout: (float) timeout for pollination reset
    :param log_level: (int) logging level
    
    """
    def __init__(self, name = None, params = None):
        
        super(ArmPlanner, self).__init__(params)
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        self._log.debug("Init Stickbug Arm Planner")        

        self._name = name
        self._params = deepcopy(params)
        
        self._state = 0
        
        self._valid_goal = False
        self._target = [0,0,0]
        self._time = 0
        
        self._support = [0,0,0]
        self._offset = self._params["support_offset"]
        if self._name[-1] == "R":
            self._offset[1] *= -1
        
        self._flowers = []
        self._pollinated = []
    
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
        # update support location and orientation

        
        if len(self._flowers) == 0:
            self._flowers = deepcopy(state["flowers"][self._name])
        else:
            temp = deepcopy(self._flowers)
            for flower in state["flowers"][self._name]:
                is_found = False
                for temp_flower in self._flowers:
                    # print(flower["position"], temp_flower["position"], np.sqrt(np.linalg.norm(np.array(flower["position"]) - np.array(temp_flower["position"]))))
                    if np.linalg.norm(np.array(flower["position"]) - np.array(temp_flower["position"])) < 5e-2:
                        is_found = True
                if not is_found:
                    temp.append(flower)
            self._flowers = temp
        
        if len(self._pollinated) == 0:
            self._pollinated = deepcopy(state["pollinated"][self._name])
        else:
            temp = deepcopy(self._pollinated)
            for flower in state["pollinated"][self._name]:
                is_found = False
                for temp_flower in self._pollinated:
                    if np.linalg.norm(np.array(flower["position"]) - np.array(temp_flower["position"])) <= self._params["pollination_radius"]:
                        is_found = True
                if not is_found:
                    temp.append(flower)
            self._pollinated = temp
        
        self._support = list(z_rotation_origin(self._offset, state["base"]["pose"][0:2] + [0], -state["base"]["pose"][3]))
        self._support.append(state["base"]["pose"][2])
        
    
class GreedyArmPlanner(ArmPlanner):
    """
    Base class for Stickbug planning
    
    **Input**
    
    :param log_level: (int) logging level
    """
    def __init__(self, name = None, params = None):
        super(GreedyArmPlanner, self).__init__(name, params)

        self._log.debug("Init Stickbug Greedy Arm Planner")
        self._pollination_time = 0        

    
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
        super().evaluate(state)
        unpollinated = []
        for flower in self._flowers:
            is_found = False
            for pol in self._pollinated:
                if np.linalg.norm(np.array(flower["position"]) - np.array(pol["position"])) <= self._params["pollination_radius"]:
                    is_found = True
            if not is_found:
                unpollinated.append(flower["position"])
                    
        print(self._name, "FLOWERS", len(self._flowers))
        print(self._name, "POLLINATED", len(self._pollinated))
        print(self._name, "UNPOLLINATED", len(unpollinated), unpollinated)
        action = {}
        
        reachable = []
        for flower in unpollinated:
            if self.constraints_satisfied(flower,state):
                reachable.append(flower)
        print(self._name, "REACHABLE", len(reachable), reachable)                  
        #need to add case for when no flowers in reach
    
        print(self._name, "Dist", np.linalg.norm(np.array(state["arms"][self._name]["position"][0:3]) - np.array(self._target)))                                         
        # level 3 competancy : go to closest flower position 
        if (self._state == "3" or self._state == "2") and np.linalg.norm(np.array(state["arms"][self._name]["position"][0:3]) - np.array(self._target)) < self._params["pollination_radius"]:
            self._pollination_time = 0
            # if self._state == "3":
            self._state = "4"
            self._valid_goal = False
            print("pollinate------------------------------------")
            return {
                        "mode": "velocity", 
                        "is_joint": False,
                        "command": [0,0, 0,0, 0,0, 0], 
                        "pollinate": True, 
                        "is_relative": False
                    }
        else:
            self._pollination_time += state["time"]-self._time
        
        check_low = self._params["l3_timeout_low"] < self._pollination_time
        print(self._name, "TIME", self._pollination_time, self._state, check_low)
        check_high = self._pollination_time < self._params["l3_timeout"] and self._state != "3"    
        if (check_low or check_high) and len(reachable):# - np.random.randint(0,25):
            temp = deepcopy(self._target)
            self._target = nearest_point(state["arms"][self._name]["position"][0:3], reachable)
            if np.linalg.norm(np.array(temp)-np.array(self._target)) > 1e-3:
                self._pollination_time = 0
            # print("--->",self._target, state["arms"][self._name]["position"][0:3])
            self._valid_goal = True #self.check_constraints(self._target,state)
            # if self._valid_goal == True:
            self._state = "3"
            print(self._name, self._valid_goal, np.linalg.norm(np.array(state["arms"][self._name]["position"][0:3]) - np.array(self._target)))
        # elif self._state == "3":
        #     self._valid_goal = False       
        
        # check_low = self._params["l3_timeout_low"] < self._pollination_time and self._state == "2"
        check_dist = np.linalg.norm(np.array(state["arms"][self._name]["position"][0:3]) - np.array(self._target)) < self._params["pollination_radius"] and self._state != "2"
        # level 2 competancy : go to a random flower position
        if check_dist or self._valid_goal == False:
            print(self._name, " RANDO-----------------------------")
            if len(reachable):
                self._target = random_point(state["arms"][self._name]["position"][0:3], reachable)
            # else:
            #     s = 1
            #     if np.random.rand() < 0.5:
            #         s = -1
            #     ind = np.random.randint(0,1)
            #     self._target = deepcopy(state["arms"][self._name]["position"][0:3])
            #     self._target[ind] += s*0.1
                self._valid_goal = True#self.check_constraints(self._target,state)
                # if self._valid_goal == True:
                self._pollination_time = 0
                self._state = "2"
            else:
                self._valid_goal = False

         # level 1 competancy : go to the rest position  
        if self._valid_goal == False and self._pollination_time < self._params["l1_timeout"]:# - random.randint(0,25):
            self._target_joints = deepcopy(self._params["joint_rest"])
            if "R" in self._name:
                self._target_joints[0] *= -1
                self._target_joints[1] *= -1
            self._state = "1"
        

        # level 0 competancy : avoid other arms in cartesian space
        force = self.obav_force(self._params["force_constant"]*np.linalg.norm(np.array(state["arms"][self._name]["position"][0:3]) - np.array(self._target)),state)
        if np.linalg.norm(force) > self._params["force_threshold"]:
            self._target = state["arms"][self._name]["position"][0:3] + force
            self._valid_goal = self.check_constraints(self._target,state)
            if self._valid_goal == True:
                self._state = "0"
                print(force)
        
        self._time = state["time"]
        
        if self._state == "1":
            action = {"mode":"position",
                      "is_joint": True,
                      "command":[0,0,state["arms"][self._name]["position"][2]] + list(self._target_joints) + [0,0],
                      "pollinate": False,
                      "is_relative": False,
                      }
        else:
            action = {"mode":"position",
                      "is_joint": False,
                      "command": list(self._target) + [0,0,0,0],
                      "pollinate": False,
                      "is_relative": False
                    }
            l = np.linalg.norm(np.array(state["arms"][self._name]["position"][0:3]) - np.array(self._target))
            temp = np.linspace(state["arms"][self._name]["position"][0:3],self._target,int(np.ceil(10*l)))
            temp = temp[0]
            if not self.check_constraints(temp,state):
                return {
                        "mode": "velocity", 
                        "is_joint": False,
                        "command": [0, 0, 0, 0, 0, 0,0], 
                        "pollinate": False, 
                        "is_relative": False
                    }
            
        return action
        
    def obav_force(self, scaling_factor=1.0,state = None):
    
        repulsive_force = np.zeros(3)
        for arm in state["arms"]:
            point = state["arms"][arm]["position"][0:3]
            # other_side = self.other_arms_sides[i]
            vector_to_point = np.array(point) - np.array(state["arms"][self._name]["position"][0:3])

            # Calculate the force magnitude inversely proportional to the distance
            distance = np.linalg.norm(vector_to_point)
            if distance != 0:
                force_magnitude = scaling_factor / (distance ** 2)
                
                # Calculate the force direction
                force_direction = -vector_to_point / distance

                # Accumulate the force
                repulsive_force += force_magnitude * force_direction 
                
        return repulsive_force
        
    def check_constraints(self, point,state):
        """
        Check if a point is within the joint constraints
        
        :param point: (list) [x,y,z] point
        :param state: (dict) dictionary of state
        :return: (bool) valid point
        """
        for key in state["arms"]:
            if "L" in key:
                # print("left arm")
                temp_keys = [el for el in state["arms"] if "L" in el and el != key]
                pts = [state["arms"][el]["position"][2] for el in temp_keys]
                pts.append(self._params["buffer"])
                pts.append(self._params["support_height"]-self._params["buffer"])
                # print(pts, point[2])
                point[2] = np.clip(point[2],self._params["buffer"]+0.0001,self._params["support_height"]-self._params["buffer"]-0.0001)
                z_max = np.min([el for el in pts if el >= point[2]])
                z_min = np.max([el for el in pts if el <= point[2]])
                # print(self._params["pose"]["left"])
            else:
                # print("right arm")
                temp_keys = [el for el in state["arms"] if "R" in el and el != key]
                pts = [state["arms"][el]["position"][2] for el in temp_keys]
                pts.append(self._params["buffer"])
                pts.append(self._params["support_height"]-self._params["buffer"])
                # print(pts, z)
                point[2] = np.clip(point[2],self._params["buffer"]+0.0001,self._params["support_height"]-self._params["buffer"]-0.0001)
                z_max = np.min([el for el in pts if el >= point[2]])
                z_min = np.max([el for el in pts if el <= point[2]])
        
        pt = np.array(point)
        support = deepcopy(self._support[0:3])
        support[2] = state["arms"][self._name]["position"][2]
        pt = z_rotation(pt,support,self._support[3]-np.pi/2) - support
        pt[2] = state["arms"][self._name]["position"][2]
        angles, valid_goal = list(arm_2d_ik(pt, self._params["mem_length"]["bicep"], self._params["mem_length"]["forearm"], "L" in self._name, state["arms"][self._name]["position"][3:6]))    
        angles = list(angles)
        if angles[1] > self._params["joint_constraints"]["th1"]["max"] or angles[1] < self._params["joint_constraints"]["th1"]["min"]:
            valid_goal = False
        if angles[2] > self._params["joint_constraints"]["th2"]["max"] or angles[2] < self._params["joint_constraints"]["th2"]["min"]:
            valid_goal = False
        
        if not valid_goal:
            return False
        
        for arm in state["arms"]:
            if arm != self._name:
                for bound in state["arms"][arm]["bounds"]:
                    if state["arms"][arm]["bounds"][bound].contains(angles):
                        print("--------------COLLISION----------------")
                        return False
        
        return valid_goal
        