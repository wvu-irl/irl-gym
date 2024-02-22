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

from irl_gym.support.stickbug.sb_base import SBBase
from irl_gym.support.stickbug.sb_support import SBSupport
from irl_gym.support.stickbug.spawn_flowers import Orchard

# For pollination should not return whether flower was polinated. Or add a flag to turn it off


class StickbugEnv(Env):
    """   
    Environment for modelling stickbug. 
    
    Due to the nature of action and observation space, env_checker has been disabled for this environment.
    
    For more information see `gym.Env docs <https://gymnasium.farama.org/api/env/>`_
        
    **States** (dict)
    
        - "base": {"pose" : [x, y, yaw], "velocity" : [v_x, v_y, v_yaw]}
        - "arms": {"<side><rel_pos>" : {"position":[x,y,z, th1, th2, cam_yaw, cam_pitch], "velocity": ..., "bounds": ...}, ...}
        
    **Observations**
    
        Agent position is fully observable {"base": {}, "arms": {}}
        "flowers": {"<side><rel_pos>" :{"position" : [x, y, z], "orientation" : [x, y, z]}}
        "pollinated": {"<side><rel_pos>" : {"position" : [x, y, z], "orientation" : [x, y, z]}}
            (Should this always return or only after a successful attempt (curretnly this one)? 
            Arguments for both...maybe add a flag)
        In the future maybe observe rows/plants too?
        time 

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
    :param render_bounds: (dict) bounds for rendering, *default*: {"x": [-5, 5], "y": [-5, 5], "z": [0, 5]}
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
        self._state = {}
        self._state, _ = self.reset(seed=seed, options=params) 
        
    def reset(self, *, seed : int = None, options : dict = {}):
        """
        Resets environment to initial state and sets RNG seed.
        
        **Deviates from Gym in that it is assumed you can reset 
        RNG seed at will because why should it matter...**
        
        :param seed: (int) RNG seed, *default*:, {}
        :param options: (dict) params for reset, see initialization, *default*: None 
        
        :return: (tuple) State Observation, Info
        """
        if not options:
            options = deepcopy(self._params)
            
        if "base" not in options:
            options["base"] = {"log_level": options["log_level"]}
            self._log.warning("No base parameters found, using defaults")
        if "support" not in options:
            options["support"] = {"log_level": options["log_level"]}
            self._log.warning("No support parameters found, using defaults")
        if "orchard" not in options:
            options["orchard"] = {"log_level": options["log_level"]}
            self._log.warning("No orchard parameters found, using defaults")
        if "row" not in options:
            options["row"] = {"log_level": options["log_level"]}
            self._log.warning("No rows parameters found, using defaults")
        if "plant" not in options:
            options["plant"] = {"log_level": options["log_level"]}
            self._log.warning("No plant parameters found, using defaults")
        if "observation" not in options:
            options["observation"] = {"log_level": options["log_level"]}
            self._log.warning("No observation parameters found, using defaults")
        if "flower" not in options:
            options["flower"] = {}
            self._log.warning("No flowers parameters found, using defaults")
        
        if "log_level" not in options["base"]:
            options["base"]["log_level"] = options["log_level"]
        if "log_level" not in options["support"]:
            options["support"]["log_level"] = options["log_level"]
        if "log_level" not in options["orchard"]:
            options["orchard"]["log_level"] = options["log_level"]
        if "log_level" not in options["row"]:
            options["row"]["log_level"] = options["log_level"]
        if "log_level" not in options["plant"]:
            options["plant"]["log_level"] = options["log_level"]
        
        if "r_range" not in options:
            options["r_range"] = (-0.01, 1)
        if "t_max" not in options:
            options["t_max"] = 100
        if "dt" not in options:
            options["dt"] = 0.1
        if "render" not in options:
            options["render"] = "plot"
        if "render_bounds" not in options:
            options["render_bounds"] = {"x": [-5, 5], "y": [-5, 5], "z": [0, 5]}
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
        self._orchard = Orchard(options["orchard"], options["row"], options["plant"])
        
        self._t = 0
        self._num_pol_prev = 0
        self._num_flowers, self._num_pollinated = self._orchard.count_flowers()
        temp = self._base.get_abs_state()
        del temp["left"]
        del temp["right"]
        self._state["base"] = temp
        self._state["arms"] = {}
        self._state["flowers"] = {}
        self._state["pollinated"] = {}
        
        if options["render"] == "plot":
            if hasattr(self, "_fig"):   
                plt.close(self._fig)
            plt.style.use('dark_background')
            self._fig = plt.figure(figsize=(10, 8), facecolor='black')
            self._ax = Axes3D(self._fig, auto_add_to_figure=False)
            self._fig.add_axes(self._ax)
            self._fn = self._fig.number
          
        # self.observation_space = spaces.discrete.Discrete(4)
        # self.action_space = spaces.Dict(
        #     {
        #         "pose": spaces.box.Box(low=np.zeros(2), high=np.array(self._params["dimensions"])-1, dtype=int)
        #     }
        # )
                
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

        if "base" not in a:
            a["base"] = {"mode": "velocity", "command": [0, 0, 0]}
        base_pose = self._base.step(a["base"], a["dt"])
        self._state["base"] = {}
        self._state["base"]["pose"] = base_pose.pop("pose")
        self._state["base"]["velocity"] = base_pose.pop("velocity")
    
        self._support.update(base_pose)
        
        obs_position = self._state["base"]["pose"][0:3]
        obs_radius = self._params["support"]["mem_length"]["bicep"] + self._params["support"]["mem_length"]["forearm"] + self._params["observation"]["camera"]["distance"]
        obs_radius += self._params["base"]["base_dim"]["radius"]
        flowers = self._orchard.get_flowers(obs_position, obs_radius)
        
        if "arms" not in a:
            a["arms"] = {}
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
        # make spit out out the bounding box on actions
        raise NotImplementedError
    
    def _get_obs(self):
        """
        Gets observation
        
        :return: (State)
        """
        self._log.debug("Get Obs: " + str(self._state))
        s = deepcopy(self._state)
        s["time"] = self._t
        return s
    
    def _get_info(self):
        """
        Gets info on system
        
        :return: (dict)
        """
        information = {"n_flowers": self._num_flowers, "n_pollinated": self._num_pollinated, "time": self._t}
        self._log.debug("Get Info: " + str(information))
        return information
    
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
            self._ax.clear()
            self._orchard.plot(self._fig,self._ax,plot=False)
            self._base.plot(self._fig,self._ax,plot=False)
            self._support.plot(self._fig,self._ax,plot=False)
            
            self._ax.set_xlabel('X-axis', color='white')
            self._ax.set_xlim(self._params["render_bounds"]["x"][0],self._params["render_bounds"]["x"][1])
            self._ax.set_ylabel('Y-axis', color='white')
            self._ax.set_ylim(self._params["render_bounds"]["y"][0],self._params["render_bounds"]["y"][1])
            self._ax.set_zlabel('Z-axis', color='white')
            self._ax.set_zlim(self._params["render_bounds"]["z"][0],self._params["render_bounds"]["z"][1])
            self._ax.set_title('Stickbug Arm Behavior Simulator', color='white')
            
            for armkey in self._state["flowers"]:
                arm = self._state["flowers"][armkey]
                for i in range(len(arm)):
                    self._ax.scatter(arm[i]["position"][0], arm[i]["position"][1], arm[i]["position"][2], color='black', marker='o')
                    
            # for armkey in self._state["pollinated"]:
            #     arm = self._state["pollinated"][armkey]
            #     if arm:
            #         self._ax.scatter(arm["position"][0], arm["position"][1], arm["position"][2], color='red', marker='o')
            
        if self._params["save_frames"]:
            plt.savefig(self._params["prefix"] + "img" + str(self._img_count) + ".png")
            self._img_count += 1
            
    def img_2_gif(self):
        """
        Converts images to gif
        """
        os.system("convert -delay 10 -loop 0 `ls -v` " + self._params["prefix"] + "img*.png " + self._params["prefix"] + "img.gif")

    def get_fignum(self):
        """
        Gets figure number
        
        :return: (int) figure number
        """
        return self._fn