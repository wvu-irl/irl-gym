"""
This module contains the StickbugEnv for simulating stickbug
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
from gymnasium import Env, spaces

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sb_base import SBBase
from sb_support import SBSupport
from sb_flowers import Orchard

class StickbugEnv(Env):
    """   
    Environment for modelling stickbug. 
    
    For more information see `gym.Env docs <https://gymnasium.farama.org/api/env/>`_
        
    **States** (dict)
    
        - "base": {"pose" : [x, y, yaw], "velocity" : [v_x, v_y, v_yaw]}
        - "arms": {"<side><rel_pos>" : {"position":[z, th1, th2, cam_yaw, cam_pitch], "velocity": ..., "bounds": ...}, ...}
        
    **Observations**
    
        Agent position is fully observable {"base": {}, "arms": {}}
        Flower positions in Observation {"position" : [x, y, z], "orientation" : [x, y, z]}
        Pollinated flowers {"<side><rel_pos>" : {"position" : [x, y, z], "orientation" : [x, y, z]}}
            (Should this always return or only after a successful attempt (curretnly this one)? 
            Arguments for both...maybe add a flag)
        In the future maybe observe rows/plants too?

    **Actions**
    
        {"base": {"mode: "position"/"velocity", "command": [x, y, yaw]},
        "arms": {"<side><rel_pos>": 
                    {
                        "mode": "position"/"velocity", 
                        "is_joint": True/False,
                        "command": [x, y, z, th1, th2, cam_yaw, cam_pitch], 
                        "pollinate": True/False, 
                        "is_relative": True/False}}
        "dt": time step}
        }
    
    **Transition Probabilities**

        Motion:
        - Base: 100 % probability of moving in commanded direction
        - Arms 100 % probability of moving in commanded direction unless boundary, in which case 0 % probability of moving in commanded direction
        - Flower Pollination: Specified by user with a pollination class
        
        
    **Reward**
    
        - $R_{min}$, cost of a single timestep (add to all time steps regardless of pollination)
        - $R_{max}$, reward for pollination all flowers
        - $R_{max}/N$, reward for pollinating a single flower, where N is the number of flowers

    
    **Input**
    
    :param seed: (int) RNG seed, *default*: None
    
    Remaining parameters are passed as arguments through the ``params`` dict.
    The corresponding keys are as follows:
    
    :param base: (dict) Parameters for the base of the stickbug
    :param support: (dict) Parameters for the support and arms of the stickbug
    :param orchard: (dict) Parameters for the orchard of flowers
    :param rows: (dict) Parameters for the rows of flowers
    :param plant: (dict) Parameters for the plants of flowers
    :param observation: (dict) Parameters for the observation of the flowers    
    :param flowers: (dict) Parameters for the flowers in the environment
    
    :param r_range: (tuple) min and max params of reward, *default*: (-0.01, 1)
    :param t_max: (int) max time steps, *default*: 100
    :param dt: (float) time step, *default*: 0.1
    :param prefix: (string) where to save images, *default*: "<cwd>/plot"
    :param render: (str) render mode, *default*: "plot"
    :param save_frames: (bool) save images for gif, *default*: False
    :param save_gif: (bool) save gif, *default*: False
    :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
    """
    metadata = {"render_modes": ["plot", "print", "none"], "render_fps": 5}

    def __init__(self, *, seed : int = None, params : dict = None):
        super(StickbugEnv, self).__init__()
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        self._log.debug("Init Stickbug Env")
        
        self._params = {}
        self._state, _ = self.reset(seed=seed, options=params) 
        
    def reset(self, *, seed : int = None, options : dict = None):
        """
        Resets environment to initial state and sets RNG seed.
        
        **Deviates from Gym in that it is assumed you can reset 
        RNG seed at will because why should it matter...**
        
        :param seed: (int) RNG seed, *default*:, {}
        :param options: (dict) params for reset, see initialization, *default*: None 
        
        :return: (tuple) State Observation, Info
        """
        
        if "base" not in options:
            options["base"] = {"log_level": options["log_level"]}
            self._log.warning("No base parameters found, using defaults")
        if "support" not in options:
            options["support"] = {"log_level": options["log_level"]}
            self._log.warning("No support parameters found, using defaults")
        if "orchard" not in options:
            options["orchard"] = {"log_level": options["log_level"]}
            self._log.warning("No orchard parameters found, using defaults")
        if "rows" not in options:
            options["rows"] = {"log_level": options["log_level"]}
            self._log.warning("No rows parameters found, using defaults")
        if "plant" not in options:
            options["plant"] = {"log_level": options["log_level"]}
            self._log.warning("No plant parameters found, using defaults")
        if "observation" not in options:
            options["observation"] = {"log_level": options["log_level"]}
            self._log.warning("No observation parameters found, using defaults")
        if "flowers" not in options:
            options["flowers"] = {}
            self._log.warning("No flowers parameters found, using defaults")
        
        if "r_range" not in options:
            options["r_range"] = (-0.01, 1)
        if "t_max" not in options:
            options["t_max"] = 100
        if "dt" not in options:
            options["dt"] = 0.1
        if "render" not in options:
            options["render"] = "plot"
        if "save_frames" not in options:
            options["save_frames"] = False
        if "prefix" not in options:
            options["prefix"] = os.getcwd() + "/plot/"
        if options["save_frames"]:
            self._img_count = 0  
            
        self._params = options
        self._seed = seed
        
        self._base = SBBase(options["base"])
        self._support = SBSupport(options["support"], options["observation"])
        self._orchard = Orchard(options["orchard"], options["rows"], options["plant"], options["flowers"])
        
        self._t = 0
        self._num_pol_prev = 0
        self._num_flowers, self._num_pollinated = self._orchard.count_flowers()
        
        self._fig = plt.figure(figsize=(10, 8), facecolor='black')
        self._ax = Axes3D(self._fig, auto_add_to_figure=False)
        self._fig.add_axes(self._ax)
        self._fn = self._fig.number
                
        return self._get_obs(), self._get_info()
    
    def step(self, a : dict = None):
        """
        Increments enviroment by one timestep 
        
        :param a: (dict) action, *default*: None
        :return: (tuple) State, reward, is_done, is_truncated, info 
        """
        self._log.debug("Step action " + str(a))
        
        done = False
        s = deepcopy(self._state) 
        
        if "dt" not in a or "dt" == -1:
            a["dt"] = self._params["dt"]

        base_pose = self._base.step(a["base"], a["dt"])
        self._state["base"]["pose"] = base_pose.pop("pose")
        self._state["base"]["velocity"] = base_pose.pop("velocity")
        
        self._support.update(base_pose)
        
        obs_position = self._state["base"]["pose"][0:3]
        obs_radius = self._params["support"]["mem_length"]["bicep"] + self._params["support"]["mem_length"]["forearm"] + self._params["observation"]["camera"]["distance"]
        obs_radius += self._params["base"]["base_dim"]["radius"]
        flowers = self._orchard.get_flowers(obs_position, obs_radius)
        
        arm_poses, arm_flowers, arm_pollinated = self._support.step(a["arms"], a["dt"], flowers)
        self._state["arms"] = deepcopy(arm_poses)
        self._state["flowers"] = deepcopy(arm_flowers)
        self._state["pollinated"] = deepcopy(arm_pollinated)
        
        r = self.reward(s, a, self._state)
        
        self._t += a["dt"]
        self._num_flowers, self._num_pollinated = self._orchard.count_flowers()
        
        if self._num_pollinated == self._num_flowers or self._t >= self._params["t_max"]:
            done = True
        
        r = self.reward(s, a, self._state)
        self._num_pol_prev = self._num_pollinated
        
        self._log.info("Is terminal: " + str(done) + ", reward: " + str(r))    
        return self._get_obs(), r, done, False, self._get_info()
    
                          
    def get_actions(self, s : dict):
        """
        Gets range of actions for a given state

        :param s: (State) state from which to get actions
        :return: (dict) Range of actions, neighbors
        """
        raise NotImplementedError
    
    def _get_obs(self):
        """
        Gets observation
        
        :return: (State)
        """
        self._log.debug("Get Obs: " + str(self._state))
        return deepcopy(self._state)
    
    def _get_info(self):
        """
        Gets info on system
        
        :return: (dict)
        """
        #probably want to get number of flowers pollinated and time ellapsed
        raise NotImplementedError
        # information = {"distance": np.linalg.norm(self._state["pose"] - self._params["goal"])}
        # self._log.debug("Get Info: " + str(information))
        # return information
    
    def reward(self, s : dict, a : int = None, sp : dict = None):
        """
        Gets rewards for $(s,a,s')$ transition
        
        :param s: (State) Initial state (unused in this environment)
        :param a: (int) Action (unused in this environment), *default*: None
        :param sp: (State) resultant state, *default*: None
        :return: (float) reward 
        """
        r = self._params["r_range"][0]*a["dt"]
        r += (self._num_pollinated - self._num_pol_prev)*self._params["r_range"][1]/self._num_flowers
        if self._num_pollinated == self._num_flowers:
            r += self._params["r_range"][1]
        return r

    def render(self):
        """    
        Renders environment
        
        Has 1 render modes: 
        
        - *plot* uses matplotlib visualization

        Visualization
        
        - ....
        """
        self._log.debug("Render " + self._params["render"])

        if self._params["render"] == "plot":
            orchard.plot(fig,ax,plot=False)
            base.plot(fig,ax,plot=False)
            support.plot(fig,ax,plot=False)

            lim = 5 # need to autogen this in the future
            
            fig.set_facecolor('black')
            ax.set_xlabel('X-axis', color='white')
            ax.set_xlim([-lim,2*lim])
            ax.set_ylabel('Y-axis', color='white')
            ax.set_ylim([-lim,2*lim])
            ax.set_zlabel('Z-axis', color='white')
            ax.set_zlim([0,lim])
            ax.set_title('Stickbug Arm Behavior Simulator', color='white')
            plt.pause(0.01)
            
        if self._params["save_frames"]:
            plt.savefig(self._params["prefix"] + "img" + str(self._img_count) + ".png")
            self._img_count += 1
            
    def img_2_gif(self):
        """
        Converts images to gif
        """
        os.system("convert -delay 10 -loop 0 " + self._params["prefix"] + "img*.png " + self._params["prefix"] + "img.gif")
