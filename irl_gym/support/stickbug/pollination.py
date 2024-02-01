"""
This module contains the base classes for pollinating flowers
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

import numpy as np

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Pollination", "PollinateRadius"]

class Pollination(ABC):
    def __init__(self, params = None):
        """
        This is the base class for pollinating flowers
        
        **Input**
        
        
        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
        """
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]                                
                                
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)
        
        self._log.debug("Init Pollination")
        
        self._params = params
        
    @abstractmethod
    def pollinate(self, position, pollinator, orientation = None):
        """
        Pollinate the flower
        
        :param position: (list) [x,y,z] position of the flower in global
        :param pollinator: (list) [x,y,z,yaw,pitch] position and orientation of the hand in global
        :param orientation: (list) [x,y,z] orientation of the flower in global
        """
        pass   
    
class PollinateRadius(Pollination):
    def __init__(self, params=None):
        """
        Pollinate a flower if it is within a certain radius
        
        **Input** 
        
        :param radius: (float) Radius to pollinate within, *default*: 0.5
        :param probability: (float) Probability of pollinating a flower, *default*: 1
        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
        """
        super().__init__(params)
        if "radius" not in params:
            self._params["radius"] = 0.5
        if "probability" not in params:
            self._params["probability"] = 1
        
    def pollinate(self, position, pollinator, orientation = None):
        """
        Pollinate the flowers
        
        :param position: (list) [x,y,z] position of the flower in global
        :param pollinator: (list) [x,y,z,yaw,pitch] position and orientation of the hand in global
        :param orientation: (list) [x,y,z] orientation of the flower in global
        :return: (bool) True if pollinated, False otherwise
        """
        if np.linalg.norm(np.array(position[0:2]) - np.array(pollinator[0:2])) < self._params["radius"]:
            if np.random.rand() < self._params["probability"]:
                return True
        return False
