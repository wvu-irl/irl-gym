"""
This module contains the model for Stickbug base
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

from collisions import BoundHexPrism

from tf import *

#later should consider modelling base more directly to capture the peculiarity of the three wheel drive

class SBBase:
    def __init__(self, params : dict = None):
        """   
        Stickbug base class for moving agent around in the environment.
        
        Note: For velocity and acceleration limits, these are assumed independent because the model is holonomic even though the robot is not.
        
        **Input**
        
        :param pose: ([x,y,z,h]) Initial pose of the agent in the environment. *default*: [0,0,0,0]
        :param velocity: (x,y,z,h) Initial velocity of the agent in the environment. *default*: [0,0,0,0]
        :param mem_offset: ([x,y,z]) Offset of the columns from the center of the agent. *default*: [0,0.5,0]
        :param max_accel: (float) Maximum acceleration of the agent. *default*: 1 (assume this includes arms and such too. Can be updated with *update_inertial_properties()*)
        :param pid: (dict) Dictionary of PID parameters for the agent. *default*: {"p": 1, "i": 0, "d": 0, "db": 0.05}
        :param pid_angular: (dict) Dictionary of PID parameters for the agent. *default*: {"p": 1, "i": 0, "d": 0, "db": 0.01}
        :param max_speed: (dict) Maximum speed of the agent. *default*: {"v": 1, "w": 1}
        :param base_dim: (dict) Radius of the base. *default*: {"radius":0.25,"height":0.5}
        
        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
        """
        super(SBBase,self).__init__()
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        self._log.debug("Init Stickbug Base")
        
        if "pose" not in params:
            params["pose"] = [0,0,0,0]
        if "velocity" not in params:
            params["velocity"] = [0,0,0,0]
        if "mem_offset" not in params:
            params["mem_offset"] = [0,0.5,0]
        if "max_accel" not in params:
            params["max_accel"] = 1
        if "pid" not in params:
            params["pid"] = {"p": 1, "i": 0, "d": 0, "db": 0.05}
        if "pid_angular" not in params:
            params["pid_angular"] = {"p": 1, "i": 0, "d": 0, "db": 0.01}
        if "max_speed" not in params:
            params["max_speed"] = {"v": 1, "w": 1}
        if "base_dim" not in params:
            params["base_dim"] = {"radius":0.25,"height":0.5}
        
        params["mem_offset"][2] += params["base_dim"]["height"]
        self._support_left = deepcopy(params["mem_offset"] )
        self._support_right = deepcopy(params["mem_offset"] )   
        self._support_right[1] *= -1
        
        self._params = params
        
        self._err_angular = {"prev": 0, "accum": 0}
        self._err = {"prev": 0, "accum": 0}
        
        self._base = BoundHexPrism(self._params["pose"][0:2], self._params["base_dim"]["radius"], self._params["base_dim"]["height"], self._params["pose"][3])
        
    def get_rel_support(self):
        """
        Get the support points of the agent.
        
        :return: ({"left":[x,y,z,h],...}) Support points of the agent.
        """
        supports = {}
        supports["left"] = deepcopy(self._support_left)
        supports["left"].append(self._params["pose"][3])
        supports["right"] = deepcopy(self._support_right)
        supports["right"].append(self._params["pose"][3])
        return supports
    
    def get_abs_support(self):
        """
        Get the support points of the agent in global frame.
        
        :param supports: ({"left":[x,y,z,h],...}) Support points of the agent. *default*: None
        :return: ({"left":[x,y,z,h],...}) Support points of the agent.
        """
        supports = self.get_rel_support()
        
        supports["left"][0:3] = z_rotation_origin(supports["left"][0:3],self._params["pose"][0:3], -self._params["pose"][3])
        supports["right"][0:3] = z_rotation_origin(supports["right"][0:3],self._params["pose"][0:3], -self._params["pose"][3])
        supports["left"][3] = self._params["pose"][3]
        supports["right"][3] = self._params["pose"][3]
        return supports
    
    def get_abs_state(self):
        """
        Gets state of base (pose, velocity) and position of supports in global frame
        
        :return: (dict) Dictionary containing the state of the agent
        """
        state = {"pose": self._params["pose"], "velocity": self._params["velocity"]}
        
        supports = self.get_abs_support()

        return {**state, **supports}


    def update_inertial_properties(self, a : float = None, alpha : float = None):
        """
        Update the inertial properties of the agent.
        
        :param a: (float) Linear acceleration of the agent. *default*: None
        :param alpha: (float) Angular acceleration of the agent. *default*: None
        """
        self._log.debug("Updating inertial properties: (a,"+str(a)+"), (alpha, "+str(alpha)+")")
        
        if a is not None:
            self._params["max_accel"] = a
    
    def step(self, action : dict = None, dt = 0.1):
        """
        Step the agent in time.
        
        :param action: (dict) Dictionary of actions to take. *default*: None
        :param dt: (float) Time step. *default*: 0.1
        :return: (dict) Dictionary containing the pose and velocity of the agent.
        """
        if action is None:
            action = {}
        if action["mode"] == "position":
            self.go_2_position(action["command"], dt)
        elif action["mode"] == "velocity":
            self.go_2_velocity(action["command"][0:2], action["command"][2], dt)
        return self.get_abs_state()   
    
    def go_2_velocity(self, v : list = None, w : float = None, dt = 0.1):
        """
        Drives base with velocity command. 
        
        :param v: ([x,y]) Linear velocity of the agent. *default*: None
        :param w: (float) Angular velocity of the agent. *default*: None
        :return: (dict) Dictionary containing the pose and velocity of the agent.
        """
        if v is None:
            v = [0,0]
        if w is None:
            w = 0
        self._log.debug("Velocity Command: (v,"+str(v)+"), (w, "+str(w)+"), ( dt, "+str(dt)+")")
        for i in range(4):
            self._params["pose"][i] += self._params["velocity"][i]*dt
        # self._params["pose"][3] += self._params["velocity"][3]*dt
        a = np.array([0,0])
        a[0] = (v[0]-self._params["velocity"][0])/dt
        a[1] = (v[1]-self._params["velocity"][1])/dt
        
        A = np.linalg.norm(a)
        if A >= self._params["max_accel"]:
            a = a*(self._params["max_accel"]/A)
        
        self._params["velocity"][0] += a[0]*dt
        self._params["velocity"][1] += a[1]*dt
        
        V = np.linalg.norm(self._params["velocity"][0:2])
        if V >= self._params["max_speed"]["v"]:
            self._params["velocity"][0] *= self._params["max_speed"]["v"]/V
            self._params["velocity"][1] *= self._params["max_speed"]["v"]/V
            
        if abs(w) >= self._params["max_speed"]["w"]:
            w *= self._params["max_speed"]["w"]/abs(w)
              
        self._params["velocity"][3] = w    
            
        self._base.update(center=self._params["pose"][0:3],heading = self._params["pose"][3])
                        
        return self.get_abs_state()
        
    
    def go_2_position(self, x : list = None, dt = 0.1):
        """
        Drives base with position command.
        
        :param x: ([x,y,h]) Position of the agent. *default*: None
        :param dt: (float) Time step. *default*: 0.1
        :return: (dict) Dictionary containing the pose and velocity of the agent.
        """
        
        err = np.array(x[0:3]) - np.array(self._params["pose"][0:3])
        err_angular = x[3] - self._params["pose"][3]
        
        v = self._params["pid"]["p"]*err + self._params["pid"]["i"]*self._err["accum"] + self._params["pid"]["d"]*(err-self._err["prev"])/dt
        w = self._params["pid_angular"]["p"]*err_angular + self._params["pid_angular"]["i"]*self._err_angular["accum"] + self._params["pid_angular"]["d"]*(err_angular-self._err_angular["prev"])/dt
        
        self._err["accum"] += err*dt
        self._err["prev"] = err
        
        self._err_angular["accum"] += err_angular*dt
        self._err_angular["prev"] = err_angular
        
        if np.linalg.norm(err) < self._params["pid"]["db"]:
            v = [0,0,0]
            # self._params["velocity"][0:3] = [0,0,0]
        if err_angular < self._params["pid_angular"]["db"]:
            w = 0
            # self._params["velocity"][3] = 0
        return self.go_2_velocity(v,w,dt)
    
    def plot(self, fig, ax, plot):
        """
        Plots the agent on an existing 3D plot.
        
        :param fig: (figure) Figure to plot on.
        :param ax: (axis) Axis to plot on.
        :param plot: (bool) Whether or not to plot.
        """
        self._base.plot(fig, ax, plot)
        
        
        
        