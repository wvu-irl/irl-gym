"""
This module contains the base class for modelling and sampling flowers
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

from pollination import *


#maintain and update age. Sample age

class Flower():
    def __init__(self, position = None, id = None, orientation = None, covariance = None, max_observations = 10, obs_update = True, pollination_params = {}):
        """
        Flower class for maintain flowers in the environment.
        
        :param position: (list) [x,y,z] position of the flower in global
        :param id: (int) Unique identifier for the flower
        :param orientation: (list) [x,y,z] orientation of the flower in global
        :param covariance: (list) Covariance of the flower in global
        :param max_observations: (int) Maximum number of observations to store
        :param obs_update: (bool) Whether to update the estimate of the flower pose when taking observations
        """
        self.id = id
        self.position = position
        if orientation is None:
            self.orientation = np.array([0,0,0])
        else:
            self.orientation = orientation
        self.covariance = covariance
        self.observations = []
        self.max_observations = max_observations
        self.obs_update = obs_update
        self.is_pollinated = False
        
        self.pollination_params = pollination_params
        if "type" not in self.pollination_params or pollination_params == {}:
            self.pollination_params["type"] = "radius"
        if self.pollination_params["type"] == "radius":
            self.pollinator = PollinateRadius(self.pollination_params)
        else:
            raise ValueError("Pollination type not recognized")
        
    def get_pose(self):
        """
        Get the pose of the flower
        
        :return: (list) [x,y,z] position of the flower in global
        """
        return self.position, self.orientation, self.covariance
    
    def get_id(self):
        """
        Get the id of the flower
        
        :return: (int) Unique identifier for the flower
        """
        return self.id
        
    def update_estimate(self, observation):
        """
        Update the estimate of the flower pose
        
        :param observation: (list) Observation of the flower
        """
        self.observations.append(observation)
        if len(self.observations) > self.max_observations:
            self.observations.pop(0)
            
        if "position" in self.observations[len(self.observations)-1]:
            self.position = self.observations[len(self.observations)-1]["position"]
        if "orientation" in self.observations[len(self.observations)-1]:
            self.orientation = self.observations[len(self.observations)-1]["orientation"]
        if "covariance" in self.observations[len(self.observations)-1]:
            self.covariance = self.observations[len(self.observations)-1]["covariance"]
            
    def sample(self):
        """
        Sample the flower pose
        """
        #
        pt = np.random.multivariate_normal(self.position, self.covariance["position"])
        orientation = np.random.multivariate_normal(self.orientation, self.covariance["orientation"])
        return pt, orientation
    
    def pollinate(self, pollinator_position):
        """
        Pollinate the flowers in the environment
        
        :param pollinator_position: (list) [x,y,z,yaw,pitch] position and orientation of the hand in global
        """
        self.is_pollinated = self.pollinator.pollinate(self.position, pollinator_position, self.orientation)

class FlowerHatchFilter(Flower):
    def __init__(self, position=None, id=None, orientation=None, covariance=None, max_observations=10):
        """
        Estimates flower locations using a hatch filter, max observations is the length of rolling average
        """
        super().__init__(position, id, orientation, covariance, max_observations)
        self.num_observations = 0
        if position is not None:
            self.num_observations += 1
        
    def update_estimate(self, observation):
        """
        Update the estimate of the flower pose as a hatch filter
        
        :param observation: (list) Observation of the flower
        """
        
        self.observations = observation
        self.num_observations += 1
        n = self.num_observations
        if n > self.max_observations:
            n = self.max_observations
        if type(self.observations) is list:
            self.observations = np.array(self.observations)
        if type(self.position) is list:
            self.position = np.array(self.position)
        self.position = (n-1)/n*self.position + 1/n*self.observations["position"]
        
        if type(self.orientation) is list:
            self.orientation = np.array(self.orientation)
        self.orientation = (n-1)/n*self.orientation + 1/n*self.observations["orientation"]
        