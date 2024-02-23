"""
This module contains the base class for observing flowers
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

from irl_gym.utils.collisions import *

#need to add noise on visibilty and position

__all__ = ["FlowerObservation", "FlowerAll", "FlowerCamera", "FlowerPyramid"]
  
class FlowerObservation(ABC):
    def __init__(self, params : dict = None):
        """   
        Flower observation class for observing flowers in the environment.
        
        **Input**
        
        :param position: (list) [x,y,z] position of the observation in global frame, *default*: [0,0,0]
        :param orientation: (list) [x,y,z] orientation of the observation in global, *default*: [0,0,0]
        :param show_camera: (bool) Whether to show the camera, *default*: False
        :param show_flowers: (bool) Whether to show the flowers in FOV, *default*: False
        :param give_id: (bool) Whether to give the flowers an id, *default*: True
        :param noise: (dict) Noise parameters {"pos_bias", [], "pos_cov": [[],[],[]], "or_cov": [[],[],[]}
        :param obs_prob: (dict) Probability of observing a flower, *default*: {"p_max": 1, "d_min": 0.1, }
        
        
        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
        """
        super(FlowerObservation,self).__init__()
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]                                
        
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)
        
        self._log.debug("Init Flower Observation")
        
        if "position" not in params:
            params["position"] = [0,0,0]
        if "orientation" not in params:
            params["orientation"] = [0,0,0]
        if "show_camera" not in params:
            params["show_camera"] = False
        if "show_flowers" not in params:
            params["show_flowers"] = False
        if "give_id" not in params:
            params["give_id"] = True
        if "noise" not in params:
            params["noise"] = {"pos_bias": [0,0,0], "pos_cov": [[0,0,0],[0,0,0],[0,0,0]], "or_cov": [[0,0,0],[0,0,0],[0,0,0]]}
        if "obs_prob" not in params:
            params["obs_prob"] = {"p_max": 1, "d_min": 0.1}
            
        self.visible_flowers = []
        
        self._params = params
        
    def update_pose(self, position = None, orientation = None):
        """
        Update the orientation of the observation
        
        :param position: (list) [x,y,z] position of the observation in global frame
        :param orientation: (list) [x,y,z] orientation of the observation in global
        """
        if position is not None:
            self._params["position"] = position
        if orientation is not None:
            self._params["orientation"] = orientation
        
    @abstractmethod
    def observe(self, flowers, position = None, orientation = None):
        """
        Observe the flowers in the environment
        
        :param flowers: (list) List of flowers in the environment
        :param position: (list) [x,y,z] position of the observation in global frame
        :param orientation: (list) [x,y,z] orientation of the observation in global
        :return: (list) List of observations
        """
        raise NotImplementedError("Please Implement this method")
    
    @abstractmethod
    def plot(self, fig, ax):
        """
        Plot the observation
        
        :param fig: (matplotlib.pyplot.figure) Figure to plot on
        :param ax: (matplotlib.pyplot.axes) Axes to plot on
        """
        raise NotImplementedError("Please Implement this method")
    
class FlowerAll(FlowerObservation):
    def __init__(self, params : dict = None):
        """
        Observe all flowers in the environment
        
        **Input**
        :param position: (list) [x,y,z] position of the observation in global frame, *default*: [0,0,0]
        :param orientation: (list) [x,y,z] orientation of the observation in global, *default*: [0,0,0]
        :param show_camera: (bool) Whether to show the camera, *default*: False
        :param show_flowers: (bool) Whether to show the flowers in FOV, *default*: False
        :param give_id: (bool) Whether to give the flowers an id, *default*: True
        
        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
        """
        super(FlowerAll,self).__init__(params)
        self.ids = 0
        if "give_id" not in params:
            self.give_id = False
        else:
            self.give_id = params["give_id"] 
        self.update_pose(params["position"], params["orientation"])
        
    def observe(self, flowers, position = None, orientation = None):
        """
        Observe the flowers in the environment
        
        :param flowers: (list) List of flowers in the environment
        :param position: (list) [x,y,z] position of the observation in global frame
        :param orientation: (list) [x,y,z] orientation of the observation in global
        :return: (list) List of observations
        """
        self.visible_flowers = []
        for flower in flowers:
            if self.give_id:
                if flower.id is None:
                    flower.id = self.ids
                    self.ids += 1
            position, orientation, _ = deepcopy(flower.get_pose())
            position = np.random.multivariate_normal(position, self._params["noise"]["pos_cov"]) + self._params["noise"]["pos_bias"]
            orientation = np.random.multivariate_normal(orientation, self._params["noise"]["or_cov"])
            self.visible_flowers.append({"position": position, "orientation": orientation})
            
        return deepcopy(self.visible_flowers)
    
    def plot(self, fig, ax):
        """
        Plot the observation
        
        :param fig: (matplotlib.pyplot.figure) Figure to plot on
        :param ax: (matplotlib.pyplot.axes) Axes to plot on
        """
        # if self._params["show_camera"]: 
        #     self.fov.plot(fig,ax,plot=False)
        
        if self._params["show_flowers"]:
            for flower in self.visible_flowers:
                ax.scatter(flower["position"][0],flower["position"][1],flower["position"][2],color='black')

    
class FlowerPyramid(FlowerObservation):
    def __init__(self, params : dict = None):
        """
        Observe all flowers in view of "pyramid" camera
        
        **Input**
        :param position: (list) [x,y,z] position of the observation in global frame, *default*: [0,0,0]
        :param orientation: (list) [x,y,z] orientation of the observation in global, *default*: [0,0,0]
        :param show_camera: (bool) Whether to show the camera, *default*: False
        :param show_flowers: (bool) Whether to show the flowers in FOV, *default*: False
        :param give_id: (bool) Whether to give the flowers an id, *default*: True
        :param camera: (dict) Camera parameters {"dist": 0.75, "spread":[0.5,0.2]}
        :param noise: (dict) Noise parameters {"pos_bias", [], "pos_cov": [[],[],[]], "or_cov": [[],[],[]}
        :param obs_prob: (dict) Probability of observing a flower, *default*: {"p_max": 1, "d_min": 0.1, }
        
        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
        """
        super(FlowerPyramid,self).__init__(params)
        self.ids = 0
        if "give_id" not in params:
            self.give_id = False
        else:
            self.give_id = params["give_id"] 
        
        self.camera = params["camera"]
        
        
    def update_pose(self, position = None, orientation = None):
        """
        Update the orientation of the observation
        
        :param position: (list) [x,y,z] position of the observation in global frame
        :param orientation: (list) [x,y,z] orientation of the observation in global
        """
        if position is not None:
            self._params["position"] = position
        if orientation is not None:
            self._params["orientation"] = orientation
        self.fov = BoundPyramid(self._params["position"], self.camera["spread"], self.camera["distance"], self._params["orientation"])
      
    def observe(self, flowers, position = None, orientation = None):
        """
        Observe the flowers in the environment
        
        :param flowers: (list) List of flowers in the environment
        :param position: (list) [x,y,z] position of the observation in global frame
        :param orientation: (list) [x,y,z] orientation of the observation in global
        :return: (list) List of observations
        """
        self.update_pose(position, orientation)
        
        temp = []

        for flower in flowers:
            position, orientation, _ = deepcopy(flower.get_pose())
            if self.fov.contains(position) and self._params["obs_prob"]["p_max"] > np.random.rand() and np.linalg.norm(np.array(position[0:2]) - np.array(self._params["position"][0:2])) > self._params["obs_prob"]["d_min"]:
                if self.give_id:
                    if flower.id is None:
                        flower.id = self.ids
                        self.ids += 1
                position = np.random.multivariate_normal(position, self._params["noise"]["pos_cov"]) + self._params["noise"]["pos_bias"]
                orientation = np.random.multivariate_normal(orientation, self._params["noise"]["or_cov"])
                temp.append({"position": position, "orientation": orientation})
                # flower.observe({"position": position, "orientation": orientation, "covariance": covariance})
        self.visible_flowers = deepcopy(temp)
        return temp
    
    def plot(self, fig, ax):
        """
        Plot the observation
        
        :param fig: (matplotlib.pyplot.figure) Figure to plot on
        :param ax: (matplotlib.pyplot.axes) Axes to plot on
        """
        if self._params["show_camera"]: 
            self.fov.plot(fig,ax,plot=False)
        
        if self._params["show_flowers"]:
            for flower in self.visible_flowers:
                ax.scatter(flower["position"][0],flower["position"][1],flower["position"][2],color='black')

class FlowerCamera(FlowerObservation):
    pass