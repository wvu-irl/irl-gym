"""
This module contains the model for Stickbug arms
"""
__license__ = "BSD-3"
__docformat__ = 'reStructuredText'
__author__ = "Trevor Smith, Jared Beard"

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from copy import deepcopy
import logging

from irl_gym.utils.collisions import BoundBox3D, BoundCylinder

import numpy as np

from irl_gym.support.stickbug.sb_arm import SBArm

class SBSupport:
    """   
    Stickbug Support class for managing movement of arms.
        
    Note: For velocity and acceleration limits, these are assumed independent because the model is holonomic even though the robot is not.
        
    **Input**
    
    :param pose: ("left": [x,y,z,h], "right": [x,y,z,h]) Initial pose of the agent in the environment. *default*: {"left": [0,0,0,0], "right": [0,0,0,0]}
    :param buffer: (float) Minimum distance between the members. *default*: 0.05
    :param support_height: (float) Height of the support. *default*: 3
    :param num_arms: (int) Number of arms on the agent. *default*: 6
    :param max_accel: (float) Maximum acceleration of the agent. *default*: 1 (assume this includes arms and such too. Can be updated with *update_inertial_properties()*)
    :param max_speed: (dict) Maximum speed of the agent. *default*: {"v": 1, "w": 1}
    :param pid: (dict) Dictionary of PID parameters for the agent. *default*: {"p": 1, "i": 0, "d": 0, "db": 0.01}
    :param pid_angular: (dict) Dictionary of PID parameters for the agent. *default*: {"p": 1, "i": 0, "d": 0, "db": 0.01}
    :param mem_length: (dict) Length of the members. *default*: {"bicep": 0.5, "forearm": 0.5}
    :param joint_rest: (list) Rest point of the joints. *default*: [0.1,0.5]
    :param joint_constraints: (dict) Constraints on the joints. *default*: {"z": {"min: <lower_joint>,"max": <upper_joint>},"th1": {"min": -np.pi, "max": np.pi}, "th2": {"min": -np.pi, "max": np.pi}}
    :param show_bounds: (bool) Whether to show the bounds of the agent. *default*: False
    
    :param observation_params: (dict) Parameters for the observation model. *default*: {"log_level": "WARNING"}
        
    :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
    """
    def __init__(self, params : dict = {}, observation_params : dict = {}):
        super(SBSupport,self).__init__()
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        self._log.debug("Init Stickbug Support")
        
        if "pose" not in params:
            params["pose"] = {"left": [0,0,0,0], "right": [0,0,0,0]}
        if "buffer" not in params:
            params["buffer"] = 0.05
        if "support_height" not in params:
            params["support_height"] = 3
        if "num_arms" not in params:
            params["num_arms"] = 6
        if "max_accel" not in params:
            params["max_accel"] = {"a": 1, "alpha": 1}
        if "max_speed" not in params:
            params["max_speed"] = {"v": 1, "w": 1}
        if "pid" not in params:
            params["pid"] = {"p": 1, "i": 0, "d": 0, "db": 0.01}
        if "pid_angular" not in params:
            params["pid_angular"] = {"p": 1, "i": 0, "d": 0, "db": 0.01}
        if "mem_length" not in params:
            params["mem_length"] = {"bicep": 0.5, "forearm": 0.5}
        if "joint_rest" not in params:
            params["joint_rest"] = [0.1,0.5]
        if "joint_constraints" not in params:
            params["joint_constraints"] = {
                "th1": {"min": -np.pi, "max": np.pi},
                "th2": {"min": -np.pi, "max": np.pi}
            }
        if "show_bounds" not in params:
            params["show_bounds"] = False
        
        if params["joint_rest"][0] > params["joint_constraints"]["th1"]["max"]:
            params["joint_rest"][0] = params["joint_constraints"]["th1"]["max"]
        elif params["joint_rest"][0] < params["joint_constraints"]["th1"]["min"]:
            params["joint_rest"][0] < params["joint_constraints"]["th1"]["min"]
        if params["joint_rest"][1] > params["joint_constraints"]["th2"]["max"]:
            params["joint_rest"][1] = params["joint_constraints"]["th2"]["max"]
        elif params["joint_rest"][1] < params["joint_constraints"]["th2"]["min"]:
            params["joint_rest"][1] < params["joint_constraints"]["th2"]["min"]
                        
                        
        self.obs_params = observation_params
        
        self._boundaries = {}
        
        self._params = params
        
        self._arms, self._arm_names = self.init_arms(params["num_arms"])
        self.update(self._params["pose"])
        
    def update(self, pose):
        """
        Update the pose of the agent.
        
        :param pose: ("left": [x,y,z,h], "right": [x,y,z,h]) Pose of the agent in the environment.
        """
        self._log.debug("Updating pose to "+str(pose))
        self._params["pose"] = deepcopy(pose)
        # self._params["pose"]["left"][3] += np.pi/2
        
        temp_bound = BoundCylinder(self._params["pose"]["left"][0:3],self._params["buffer"], self._params["support_height"],end_sphere=True)
        self._boundaries = {**self._boundaries, **{"col_l":temp_bound}}
        temp_bound = BoundCylinder(self._params["pose"]["right"][0:3],self._params["buffer"], self._params["support_height"],end_sphere=True)
        self._boundaries = {**self._boundaries, **{"col_r":temp_bound}}
        
        midpoint = (np.array(self._params["pose"]["left"][0:3]) + np.array(self._params["pose"]["right"][0:3])) / 2
        dist = np.linalg.norm(np.array(self._params["pose"]["left"][0:2]) - np.array(self._params["pose"]["right"][0:2]))
        dist += np.sum([el for el in self._params["mem_length"].values()])
        self._boundaries = {**self._boundaries, **{"floor":BoundCylinder(midpoint+[0,self._params["buffer"],0], 2*dist, 2*self._params["support_height"],np.pi)}}
        self._boundaries = {**self._boundaries, **{"ceiling":BoundCylinder(midpoint+[0,-self._params["buffer"],self._params["support_height"]], 2*dist, 2*self._params["support_height"])}}

        temp_bounds = {}
        for key in self._arms:
            temp_bounds[key] = deepcopy(self._boundaries)

        for key in self._arms:
            for key2 in self._arms:
                if key != key2:
                    temp_bounds = {**temp_bounds, **(self._arms[key2].get_bounds())}

        # print("checking available bounds")
        for key in self._arms:
            arm_bounds = {"joints": deepcopy(self._params["joint_constraints"]), "bounds": temp_bounds[key]}
            arm_bounds["joints"]["z"] = {}
            if "L" in key:
                # print("left arm")
                temp_keys = [el for el in self._arms if "L" in el and el != key]
                pts = [self._arms[el].get_absolute_state() for el in temp_keys]
                pts = [el["position"][2] for el in pts]
                pts.append(-self._params["buffer"])
                pts.append(self._params["support_height"]-self._params["buffer"])
                z = self._arms[key].get_absolute_state()["position"][2]
                # print(pts, z)
                arm_bounds["joints"]["z"]["max"] = np.min([el for el in pts if el >= z])
                arm_bounds["joints"]["z"]["min"] = np.max([el for el in pts if el <= z])
                # print(self._params["pose"]["left"])
                self._arms[key].update(shoulder_pose = self._params["pose"]["left"], constraints = arm_bounds)
            else:
                # print("right arm")
                temp_keys = [el for el in self._arms if "R" in el and el != key]
                pts = [self._arms[el].get_absolute_state() for el in temp_keys]
                pts = [el["position"][2] for el in pts]
                pts.append(-self._params["buffer"])
                pts.append(self._params["support_height"]-self._params["buffer"])
                # print(pts, z)
                z = self._arms[key].get_absolute_state()["position"][2]
                arm_bounds["joints"]["z"]["max"] = np.min([el for el in pts if el >= z])
                arm_bounds["joints"]["z"]["min"] = np.max([el for el in pts if el <= z])
                self._arms[key].update(shoulder_pose = self._params["pose"]["right"], constraints = arm_bounds)     
        
    def init_arms(self, num: int = 6):
        """
        Initialize the arms on the agent.
        
        :param num: (int) Number of arms to initialize. *default*: 6
        :returns: (list) List of arms
        """
        self._log.debug("Initializing " + str(num) + " arms")
        naming = ["T", "M", "B", "BB", "BBB"]
        
        if num % 2 != 0:
            l_arm = num // 2
            r_arm = num // 2 + 1
        else:
            l_arm = num // 2
            r_arm = num // 2
        
        arm_params = {
            "max_accel": self._params["max_accel"],
            "max_speed": self._params["max_speed"],
            "pid": self._params["pid"],
            "pid_angular": self._params["pid_angular"],
            "mem_length": self._params["mem_length"],
            "buffer": self._params["buffer"],
            "show_bounds": self._params["show_bounds"],
            "log_level": self._params["log_level"],
        }
        
        left_gap = self._params["support_height"]# - 2*self._params["buffer"]
        right_gap = left_gap
        
        left_dist = left_gap / (l_arm + 1)
        right_dist = right_gap / (r_arm + 1)
                        
        arms = {}
        arm_names = []

        arm_params["pose"] = {}
        for i in range(l_arm):
            arm_params["location"] = {"column": "L", "level": naming[i]}
            arm_params["pose"]["linear"] = self._params["pose"]["left"][0:3]
            arm_params["pose"]["linear"][2] += left_dist*(l_arm-i)
            arm_params["pose"]["angular"] = [self._params["pose"]["left"][3], self._params["joint_rest"][0], self._params["joint_rest"][1], 0,0]
            arms = {**arms, str(naming[i])+"L":SBArm(arm_params, deepcopy(self.obs_params))}
            arm_names.append(str(naming[i])+"L")
        
        for i in range(r_arm):
            arm_params["location"] = {"column": "R", "level": naming[i]}
            arm_params["pose"]["linear"] = self._params["pose"]["right"][0:3]
            arm_params["pose"]["linear"][2] += right_dist*(r_arm-i)
            arm_params["pose"]["angular"] = [self._params["pose"]["right"][3], -self._params["joint_rest"][0], -self._params["joint_rest"][1],0,0]
            arms = {**arms, str(naming[i])+"R":SBArm(arm_params, deepcopy(self.obs_params))} 
            arm_names.append(str(naming[i])+"R")
            
        return arms, arm_names
        
    def step(self, action : dict = {}, dt = 0.1, flowers = []):
        """
        Step the agent forward in time.
        
        :param action: (dict) Action to take. *default*: {}}
        :param dt: (float) Time step to use. *default*: 0.1
        :param flowers: (list) List of flowers in the environment. *default*: []
        """
        self._log.debug("Stepping")
        
        for key in self._arm_names:
            if key not in action:
                action[key] = {}
            
        arm_poses = {}
        arm_flowers = {}
        arm_pollinated = {}
        for key in action:
            act = action[key]
            act["dt"] = dt
            pos, obs_flowers, pol_flowers = self._arms[key].step(act, flowers)
            arm_poses[key] = pos
            arm_flowers[key] = obs_flowers
            arm_pollinated[key] = pol_flowers
        
        self.arm_flowers = arm_flowers
        self.update(self._params["pose"])
        return arm_poses, arm_flowers, arm_pollinated
        
    def observe_flowers(self, flowers = None):
        """
        Observe the flowers in the environment
        
        :param flowers: (list) List of flowers in the environment
        :return: (list) List of observations for each arm
        """
        if flowers is None:
            return deepcopy(self.arm_flowers)
        arm_flowers = {}
        for key in self._arms:
            arm_flowers[key] = self._arms[key].observe_flowers(flowers)
        return deepcopy(arm_flowers)
        
    def plot(self, fig, ax, plot):
        """
        Plot the agent in the environment.
        
        :param fig: (matplotlib.figure.Figure) Figure to plot on
        :param ax: (matplotlib.axes.Axes) Axes to plot on
        :param plot: (bool) Whether to plot the agent
        """
        pt = deepcopy(self._params["pose"]["left"])
        pt[2] = self._params["support_height"]
        ax.plot([self._params["pose"]["left"][0], pt[0]], [self._params["pose"]["left"][1], pt[1]], [self._params["pose"]["left"][2], pt[2]], 'go-', lw=1)  # Vertical line to base
        pt = deepcopy(self._params["pose"]["right"])
        pt[2] = self._params["support_height"]
        ax.plot([self._params["pose"]["right"][0], pt[0]], [self._params["pose"]["right"][1], pt[1]], [self._params["pose"]["right"][2], pt[2]], 'go-', lw=1)  # Vertical line to base
        
        if self._params["show_bounds"]:
            for key, bnd in self._boundaries.items():
                bnd.plot(fig, ax, plot)
        
        for key in self._arms:
            self._arms[key].plot_arm(fig, ax, plot)