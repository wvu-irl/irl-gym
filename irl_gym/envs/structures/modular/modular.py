"""
This module contains the Assumption object for controlling how assumptions 
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

__all__ = ["Assumption", "check_limits"] 

# Modular environment
# Assume some assumption {"type": {"name", <name>, "prop1": <>, ...}}
# Then look up function based on type and name
# Because assumptions updated dynamically, need a reconfigure option

# limits and initial state should have same structure. 
#Then for each value of limits dict, if value is not satisfied, raise error with message
# Else, if during update and value is not satisfied, do nothing or select nearest value (depends on whether or not numeric)





class Assumption:
    """
    Assumption object for controlling assumption properties
    """
    def __init__(self, name = str, initial_state = None, limits = None):
        self.__name = name
        
        # Raise Error if none?
        self.__state = initial_state
        self.__limits = limits
        
        if not self.check_limits():
            raise ValueError("Initial state does not satisfy limits")
        
    def check_limits(self, value = None):
        """_summary_

        Args:
            limits (_type_): _description_
            value (_type_): _description_

        Returns:
            _type_: _description_
        """
        if value is None:
            value = self.__state
        
        for key in self.__limits:
            if key not in value or value[key] is None:
                return False
            for i, el in enumerate(self.__limits[key]):
                if type(el) is list:
                    if value[key] >= el[0] and value[key] <= el[1]:
                        return True
                elif value[key] == el:
                    return True
        return False
        
    def compute(self, s1, s2 = None):
        """
        Compute the assumption value
        """
        raise NotImplementedError
    
    def update(self, state):
        """
        Update the assumption value
        """
        raise NotImplementedError
    
    def check_validity(self, data):
        """
        Check if the assumption is valid
        
        
        """
        raise NotImplementedError
    
    