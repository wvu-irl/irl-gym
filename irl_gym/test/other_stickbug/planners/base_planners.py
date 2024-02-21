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

from irl_gym.test.other_stickbug.planners.planner import Planner

class BasePlanner(Planner):
    """
    Base class for Stickbug planning
    
    **Input**
    
    :param log_level: (int) logging level
    :param base_params: (dict) dictionary of base parameters
    :param arm_params: (dict) dictionary of arm parameters
    """
    def __init__(self, params = None):
        
        super(BasePlanner, self).__init__(params)
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        self._log.debug("Init Stickbug Base Planner")
        
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
        raise NotImplementedError("evaluate not implemented")
    
def BaseWaypointsPlanner(BasePlanner):
    """
    Performs planning for a set of provided waypoints
    
    **Input**
    
    :param waypoints: (list) list of waypoints to follow [[x,y,h],...]
    :param log_level: (int) logging level
    """
    def __init__(self, params = None):
        
        super(BaseWaypointsPlanner, self).__init__(params)
        
        self._waypoints = params["waypoints"]
        self._waypoint_idx = 0
        
    def reinit(self, state = None, action = None, s_prime = None):
        """
        Reinitialize Planner
        
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
        d = np.linalg.norm(np.array(self._waypoints[self._waypoint_idx]) - np.array(state["base"]["position"]))
        if self._waypoint_idx >= len(self._waypoints):
            return {"mode":"position",
                    "command":self._waypoints[-1]}
        else:
            if d < self._params["waypoint_threshold"]:
                self._waypoint_idx += 1
            return {"mode":"position",
                    "command":self._waypoints[self._waypoint_idx]}