"""
This module contains the base classes for spawning flowers
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

from irl_gym.support.stickbug.flowers import Flower
from irl_gym.utils.collisions import *

__all__ = ["Flower", "Cluster", "Plant", "Row", "Orchard"]

#should separate our params from sub params so they can be passed to sub classes

class Cluster():
    def __init__(self,params = None, flower_params = None):
        """
        This class generates a cluster of flowers
        
        **Input**
        
        :param position: (list) [x,y,z] position of the cluster in global frame
        :param radius: (float) Radius of the cluster, *default*: 0.1
        :param num_flowers: (int) Number of flowers in the cluster, *default*: 5
        :param cluster_covariance: (list) Covariance of the cluster, if empty -> uniform sample, *default*: []
        :param random_size: (bool) Whether to randomly sample the size of the cluster, if True -> radius is max_size, *default*: False
        :param random_num: (bool) Whether to randomly sample the number of flowers in the cluster, if True -> num_flowers is max_num, *default*: False
        :param show_cluster: (bool) Whether to show the cluster, *default*: False
        
        :param flower_params: (dict) Parameters for the flowers {"flower": {}}
        
        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING
        """
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]                                
        
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)
        
        self._log.debug("Init Cluster")
        
        if "position" not in params:
            params["position"] = [0,0,0]
        if "radius" not in params:
            params["radius"] = 0.1
        if "num_flowers" not in params:
            params["num_flowers"] = 5
        if "cluster_covariance" not in params:
            params["cluster_covariance"] = []
        if "random_size" not in params:
            params["random_size"] = False
        if "random_num" not in params:
            params["random_num"] = False
        if "show_cluster" not in params:
            params["show_cluster"] = False
            
        if flower_params is None:
            raise ValueError("flower_params must be defined")
        
        params["flower_params"] = flower_params
        
        self._params = params
        
        self.generate_flowers()
        
    def generate_flowers(self):
        flowers = []
        if self._params["random_size"]:
            self._params["radius"] = np.random.uniform(0,self._params["radius"])
        if self._params["random_num"]:
            self._params["num_flowers"] = np.random.randint(1,self._params["num_flowers"])
        
        while len(flowers) < self._params["num_flowers"]:
            if len(self._params["cluster_covariance"]) == 0:
                pos = np.random.uniform(-self._params["radius"],self._params["radius"],3)
            else:
                pos = np.random.multivariate_normal([0,0,0],self._params["cluster_covariance"])
            if np.linalg.norm(pos) <= self._params["radius"]:
                pos += self._params["position"]
                self._params["flower_params"]["position"] = pos
                flowers.append(Flower(**self._params["flower_params"]))
        self.flowers = flowers
    
    def get_flowers(self, point = None, radius = None):
        """
        Get the flowers in the cluster
        
        :param point: (list) [x,y,z] position of the point to get flowers around, if None -> get all flowers, *default*: None
        :param radius: (float) Radius to get flowers around, if None -> get all flowers, *default*: None
        :return: (list) List of flowers in the cluster
        """
        if point is None or radius is None:
            return self.flowers
        
        flowers = []
        for flower in self.flowers:
            dist = np.linalg.norm(np.array(flower.position)-np.array(point))
            if dist <= radius:
                flowers.append(flower)
        return flowers
    
    def plot(self, fig, ax, plot = False):
        """
        Plot the cluster
        
        :param fig: (matplotlib.pyplot.figure) Figure to plot on
        :param ax: (matplotlib.pyplot.axis) Axis to plot on
        :param plot: (bool) Whether to plot or not, *default*: False
        """
        for flower in self.flowers:
            ax.scatter(flower.position[0],flower.position[1],flower.position[2],color = "c")
        if self._params["show_cluster"]:
            BoundSphere(self._params["position"], self._params["radius"]).plot(fig, ax, plot)
            
    def count_flowers(self):
        """
        Count the number of flowers in the cluster
        
        :return: (int) Number of flowers in the cluster, number pollinated
        """
        total = len(self.flowers)
        total_pollinated = 0
        for flower in self.flowers:
            if flower.is_pollinated:
                total_pollinated += 1
        return total, total_pollinated

class Plant():
    def __init__(self,params = None, cluster_params = None, flower_params = None):
        """
        This class generates a plant with clusters of flowers and flowers
        
        **Input**
        
        :param position: (list) [x,y,z] position of the plant in global frame
        :param height: (float) height of the plant
        :param radius: (float) Radius of the plant, *default*: 0.3
        :param is_type: (str) Type of plant (cylinder or cone), *default*: "cylinder"
        :param num_clusters: (int) Number of clusters in the plant, if None -> fit to constraining dimension, *default*: None
        :param show_plant: (bool) Whether to show the plant, *default*: False

        :param cluster_params: (dict) Parameters for the clusters {"num_clusters": 1, "strict_bounds": False, "random_num" = True, "cluster": {}}

        :param flower_params: (dict) Parameters for the flowers {"num_flowers": 1, "random_num" = True, "flower": {}}

        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
        """
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]                                
        
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)
        
        self._log.debug("Init Plant")
        
        if "position" not in params:
            params["position"] = [0,0,0]
        if "height" not in params:
            params["height"] = 1
        if "radius" not in params:
            params["radius"] = 0.3
        if "is_type" not in params:
            params["is_type"] = "cylinder"
        if "show_plant" not in params:
            params["show_plant"] = False
            
        if cluster_params is None and flower_params is None:
            raise ValueError("cluster and/or flowers params must be defined")
        if cluster_params is None or len(cluster_params) == 0:
            params["cluster"] = {}
        else:
            params["cluster"] = cluster_params
        if flower_params is None or len(flower_params) == 0:
            params["flower"] = {}
        else:
            params["flower"] = flower_params
        
        self._params = params
        self.clusters = []
        self.flowers = []
        
        if params["cluster"]["num_clusters"]:
            self.generate_clusters()
        if params["flower"]["num_flowers"]:
            self.generate_flowers()
        
    def generate_clusters(self):
        """
        Generates clusters of flowers
        """
        if "random_num" not in self._params["cluster"]:
            self._params["cluster"]["random_num"] = True    
        if self._params["cluster"]["random_num"] > 1:
            self._params["cluster"]["num_clusters"] = np.random.randint(1,self._params["cluster"]["num_clusters"])
            
        for i in range(self._params["cluster"]["num_clusters"]):
            #taking shortcut here with hard limit. Assuming that in case of random sampling then max radius should still be used
            sample_min = deepcopy(self._params["position"])
            radius = deepcopy(self._params["radius"])
            height = deepcopy(self._params["height"])
            if self._params["is_type"] == "cylinder":
                if "strict_bounds" in self._params["cluster"] and self._params["cluster"]["strict_bounds"]:
                    sample_min[2] += self._params["cluster"]["cluster"]["radius"]
                    height -= self._params["cluster"]["cluster"]["radius"]
                    radius -= self._params["cluster"]["cluster"]["radius"]
                
                theta = np.random.uniform(0,2*np.pi)
                z = np.random.uniform(sample_min[2],sample_min[2]+height)
                self._params["cluster"]["cluster"]["position"] = [sample_min[0]+radius*np.cos(theta),sample_min[1]+radius*np.sin(theta),z]
            
            else:
                radius = deepcopy(self._params["radius"])
                if "strict_bounds" in self._params["cluster"] and self._params["cluster"]["strict_bounds"]:
                    sample_min[2] += self._params["cluster"]["cluster"]["radius"]
                    height -= height*self._params["cluster"]["cluster"]["radius"]/radius
                    radius -= self._params["cluster"]["cluster"]["radius"]
                
                theta = np.random.uniform(0,2*np.pi)
                z = np.random.uniform(sample_min[2],sample_min[2]+height)
                r = radius*(1-z/height)
                self._params["cluster"]["cluster"]["position"] = [sample_min[0]+r*np.cos(theta),sample_min[1]+r*np.sin(theta),z]
            
            cluster = Cluster(deepcopy(self._params["cluster"]["cluster"]), deepcopy(self._params["flower"]["flower"]))
            self.clusters.append(cluster)
        
    def generate_flowers(self):
        """
        Generates flowers
        """        
        if "random_num" not in self._params["flower"]:
            self._params["flower"]["random_num"] = True
        if self._params["flower"]["random_num"]:
            self._params["flower"]["num_flowers"] = np.random.randint(1,self._params["flower"]["num_flowers"])
            
        temp = deepcopy(self._params["flower"]["flower"])
        for i in range(self._params["flower"]["num_flowers"]):
            theta = np.random.uniform(0,2*np.pi)
            z = np.random.uniform(self._params["position"][2],self._params["position"][2]+self._params["height"])
            temp["position"] = [self._params["position"][0]+self._params["radius"]*np.cos(theta),self._params["position"][1]+self._params["radius"]*np.sin(theta),z]
            self.flowers.append(Flower(**temp))
            
    def get_flowers(self, point = None, radius = None):
        """
        Get the flowers in the plant
        
        :param point: (list) [x,y,z] position of the point to get flowers around, if None -> get all flowers, *default*: None
        :param radius: (float) Radius to get flowers around, if None -> get all flowers, *default*: None
        :return: (list) List of flowers in the plant
        """
        flowers = []
        if point is None or radius is None:
            flowers.extend(self.flowers) 
        else:
            for flower in self.flowers:
                dist = np.linalg.norm(np.array(flower.position)-np.array(point))
                if dist <= radius:
                    flowers.append(flower)
                    
        for cluster in self.clusters:
                flowers.extend(cluster.get_flowers(point, radius))
        
        return flowers
    
    def get_pose(self):
        """
        Get the pose of the plant
        
        :return: (list) [x,y,z] position of the plant in global
        """
        return self._params["position"]
    
    def plot(self, fig, ax, plot = False):
        """
        Plot the plant
        
        :param fig: (matplotlib.pyplot.figure) Figure to plot on
        :param ax: (matplotlib.pyplot.axis) Axis to plot on
        :param plot: (bool) Whether to plot or not, *default*: False
        """
        for flower in self.flowers:
            ax.scatter(flower.position[0],flower.position[1],flower.position[2],color = "white")
        
        for cluster in self.clusters:
            cluster.plot(fig, ax, plot)
        
        if self._params["show_plant"]:
            if self._params["is_type"] == "cylinder":
                BoundCylinder(self._params["position"], self._params["radius"], self._params["height"]).plot(fig, ax, plot,"g")
            else:
                BoundCone(self._params["position"], self._params["radius"], self._params["height"]).plot(fig, ax, plot, "g")    
                
    def count_flowers(self):
        """
        Count the number of flowers in the plant
        
        :return: (int) Number of flowers in the plant, number pollinated
        """
        total = len(self.flowers)
        total_pollinated = 0
        for flower in self.flowers:
            if flower.is_pollinated:
                total_pollinated += 1
                
        for cluster in self.clusters:
            temp, temp_p = cluster.count_flowers()
            total += temp
            total_pollinated += temp_p
        return total, total_pollinated

class Row():
    def __init__(self,params = None, plant_params = None):
        """
        This class generates a row of flowers
        
        **Input**
        
        :param position: (list) [x,y,z] position of the row in global
        :param size: (list) [x,y,z] size of the row
        :num_plants: (int) Number of plants in the row, if None -> fit to constraining dimension, *default*: None
        
        :param plant_params: (dict) Parameters for the plants
        
        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
        """
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]                                
        
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)
        
        self._log.debug("Init Row")
        
        if "position" not in params:
            params["position"] = [0,0,0]
        if "size" not in params:
            params["size"] = [1,1,1]
        if "num_plants" not in params:
            params["num_plants"] = None
            
        if plant_params is None:
            raise ValueError("plant_params must be defined")
        params["plant"] = plant_params
        
        self._params = params
        
        self.plants = []
        self.generate_plants()
        
    def generate_plants(self):
        """
        Generates the row of plants
        """
        if self._params["num_plants"] is None:
            self._params["num_plants"] = int(np.floor(np.max(self._params["size"][0:2])/(2*self._params["plant"]["plant"]["radius"])))
            
        r = np.max(self._params["size"][0:2])/self._params["num_plants"]
        for i in range(self._params["num_plants"]):
            if self._params["size"][0] > self._params["size"][1]:
                x = self._params["position"][0]+r*(i+1/2)  
                y = self._params["position"][1]+ self._params["size"][1]/2
            else:
                x = self._params["position"][0]+ self._params["size"][0]/2
                y = self._params["position"][1]+r*(i+1/2)
            z = self._params["position"][2]+ self._params["size"][2]
            self._params["plant"]["plant"]["position"] = [x,y,z]
            self.plants.append(Plant(deepcopy(self._params["plant"]["plant"]), deepcopy(self._params["plant"]["cluster"]), deepcopy(self._params["plant"]["flower"])))
    
    def get_flowers(self,point = None, radius = None):
        """
        Get the flowers in the row
        
        :param point: (list) [x,y,z] position of the point to get flowers around, if None -> get all flowers, *default*: None
        :param radius: (float) Radius to get flowers around, if None -> get all flowers, *default*: None
        :return: (list) List of flowers in the row
        """
        flowers = []
        for plant in self.plants:
            if point is None or radius is None:
                flowers.extend(plant.get_flowers())
            else:
                dist = np.linalg.norm(np.array(plant.get_pose()[0])-np.array(point))
                if dist <= radius:
                    flowers.extend(plant.get_flowers())
                
        return flowers
    
    def plot(self, fig, ax, plot = False):
        """
        Plot the plant
        
        :param fig: (matplotlib.pyplot.figure) Figure to plot on
        :param ax: (matplotlib.pyplot.axis) Axis to plot on
        :param plot: (bool) Whether to plot or not, *default*: False
        """
        BoundRectPrism(self._params["position"], self._params["size"]).plot(fig, ax, plot)

        for plant in self.plants:
            plant.plot(fig, ax, plot)     
            
    def count_flowers(self):
        """
        Count the number of flowers in the row
        
        :return: (int) Number of flowers in the row, number pollinated
        """
        total = 0
        total_pollinated = 0
        for plant in self.plants:
            temp, temp_pollinated = plant.count_flowers()
            total += temp
            total_pollinated += temp_pollinated
        return total , total_pollinated 

class Orchard():
    def __init__(self,params=None, row_params = None, plant_params = None):
        """
        Generates an orchard of plants
        
        **Input**
        
        :param offset: (list) [x,y,z] offset of the orchard in global
        :param num_rows: (int) Number of rows in the orchard, *default*: 1
        :param spacing: (float) Spacing between rows, *default*: 1
        :param is_double: (bool) Whether to have double rows, if True -> num_rows is num of double rows, *default*: False
        
        :param row_params: (dict) Parameters for the rows
        :param plant_params: (dict) Parameters for the plants
        """
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]                                
        
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)
        
        self._log.debug("Init Orchard")
        
        if "offset" not in params:
            params["offset"] = [0,0,0]
        if "num_rows" not in params:
            params["num_rows"] = 1
        if "spacing" not in params:
            params["spacing"] = 1
        if "is_double" not in params:
            params["is_double"] = False

        if row_params is None and plant_params is None or row_params is {} and plant_params is {}:
            raise ValueError("row and/or plants params must be defined")
        if row_params is None or len(row_params) == 0:
            params["row"] = {}
        else: 
            params["row"] = row_params
        if plant_params is None or len(plant_params) == 0:
            params["plant"] = None
        else:
            params["plant"] = plant_params
        
        self._params = params
        
        self.rows = []
        self.generate_orchard()
        
    def generate_orchard(self):
        """
        Generates the orchard
        """
        if not self._params["row"]:
            self._params["row"]["size"] = [2*self._params["plant"]["plant"]["radius"],2*self._params["plant"]["plant"]["radius"],0]
            self._params["row"]["num_plants"] = 1
            
        if self._params["row"]["size"][0] > self._params["row"]["size"][1]:
            space = np.array([0,self._params["spacing"],0])
        else:
            space = np.array([self._params["spacing"],0,0])
            
        row_pos = np.array(deepcopy(self._params["offset"]))
        
        for i in range(self._params["num_rows"]):
            self._params["row"]["position"] = row_pos
            self.rows.append(Row(deepcopy(self._params["row"]), deepcopy(self._params["plant"])))
                
            if self._params["is_double"]:
                if self._params["row"]["size"][0] > self._params["row"]["size"][1]:
                    row_pos[1] += self._params["row"]["size"][1]
                else:
                    row_pos[0] += self._params["row"]["size"][0]
                self._params["row"]["position"] = row_pos
                self.rows.append(Row(deepcopy(self._params["row"]), deepcopy(self._params["plant"])))
                # row_pos += [self._params["row"]["size"][0],self._params["row"]["size"][1],0]
            
            if self._params["row"]["size"][0] > self._params["row"]["size"][1]:
                row_pos[1] += self._params["row"]["size"][1]
            else:
                row_pos[0] += self._params["row"]["size"][0]
            row_pos += space    
    
    def get_flowers(self, position = None, radius = None):
        """
        Get the flowers in the orchard
        
        :param position: (list) [x,y,z] position of the point to get flowers around, if None -> get all flowers, *default*: None
        :param radius: (float) Radius to get flowers around, if None -> get all flowers, *default*: None
        """
        flowers = []
        for row in self.rows:
            flowers.extend(row.get_flowers(position, radius))
        return flowers
    
    def plot(self, fig, ax, plot = False):
        """
        Plot the plant
        
        :param fig: (matplotlib.pyplot.figure) Figure to plot on
        :param ax: (matplotlib.pyplot.axis) Axis to plot on
        :param plot: (bool) Whether to plot or not, *default*: False
        """
        for row in self.rows:
            row.plot(fig, ax, plot)
            
    def count_flowers(self):
        """
        Count the number of flowers in the orchard
        
        :return: (int) Number of flowers in the orchard, number of flowers pollinated
        """
        total = 0
        total_pollinated = 0
        for row in self.rows:
            temp, temp_pol = row.count_flowers()
            total += temp
            total_pollinated += temp_pol
            
        return total, total_pollinated
        

#class spawn flowers in area
    # 2 versions, can do rectangles
    # can do rectangles with cylinders


#need to parameterize camera params for arm and put in observation model...
# perhaps color flowers in arm slightly differentf