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

from irl_gym.utils.collisions import *
from irl_gym.utils.tf import *
from irl_gym.support.stickbug.flower_observation import *
from irl_gym.test.other_stickbug.planners.plan_utils import arm_2d_ik

import numpy as np

#need to import class of flower representations.
# need to work on manipulation of two end joints for observations

class SBArm:
    """   
        Stickbug Arm class for moving arms on the robot.
        
        Rest point is set to initial pose Unless obstructed, then z is fixed.
        
        Note: For velocity and acceleration limits, these are assumed independent because the model is holonomic even though the robot is not.
        
        **Input**
        
        :param location: (dict) Which column arm is on, and if upper, middle, lower
        :param pose: (dict) Initial pose of the arm. *default*: {"linear":[0,0,0], "angular":[0,0,0,0,0]]} # col angle, th1, th2, yaw,pitch
        :param velocity: ([v,w1,w2]) Initial velocity of the arm. *default*: [0,0,0]
        :param max_accel: (float) Maximum acceleration of the agent. *default*: 1
        :param max_speed: (dict) Maximum speed of the agent. *default*: {"v": 1, "w": 1}
        :param pid: (dict) Dictionary of PID parameters for the agent. *default*: {"p": 1, "i": 0, "d": 0, "db": 0.01}
        :param pid_angular: (dict) Dictionary of PID parameters for the agent. *default*: {"p": 1, "i": 0, "d": 0, "db": 0.01}
        :param mem_length: (dict) Length of the members. *default*: {"bicep": 0.5, "forearm": 0.5}
        :param rest_point: (list) Rest point of the arm. *default*: [0,0,0]
        :param buffer: (float) Buffer for the arm. *default*: 0.1
        :param joint_constraints: (dict) Constraints on the joints. *default*: {"z": {"min: <lower_joint>,"max": <upper_joint>},"th1": {"min": -np.pi, "max": np.pi}, "th2": ..., "yaw": ..., "pitch": ...}
        :param show_bounds: (bool) Whether to show the bounds of the agent. *default*: False
        :param show_camera: (bool) Whether to show the camera of the agent. *default*: False

        :param observation_params: (dict) Parameters for observing flowers. *default*: {"position": [0,0,0], "orientation": [0,0,0]}
        
        :param log_level: (str) Level of logging to use. For more info see `logging levels <https://docs.python.org/3/library/logging.html#levels>`_, *default*: "WARNING"
    """
    def __init__(self, params : dict = {}, observation_params : dict = {}):
        super(SBArm,self).__init__()
                 
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        if "location" not in params:
            raise ValueError("Arm location not specified")
        
        self._log.debug("Init Stickbug Arm "+params["location"]["level"]+params["location"]["column"])

        if "pose" not in params:
            params["pose"] = {"linear":[0,0,0], "angular":[0,0,0,0,0]} # col angle, th1, th2, yaw,pitch
        self._rest = params["pose"]
        if "velocity" not in params:
            params["velocity"] = {"linear":[0,0,0], "angular":[0,0,0,0,0]}
        if "max_accel" not in params:  
            params["max_accel"] = 1
        if "max_speed" not in params:
            params["max_speed"] = {"v": 1, "w": 1}
        if "pid" not in params:
            params["pid"] = {"p": 1, "i": 0, "d": 0, "db": 0.05}
        if "pid_angular" not in params:
            params["pid_angular"] = {"p": 1, "i": 0, "d": 0, "db": 0.01}
        if "mem_length" not in params:
            params["mem_length"] = {"bicep": 0.5, "forearm": 0.5}
        if "buffer" not in params:
            params["buffer"] = 0.1
        if "joint_constraints" not in params:
            params["joint_constraints"] = {"z": {"min": 0,"max": 2},"th1": {"min": -np.pi, "max": np.pi}, "th2": {"min": -np.pi, "max": np.pi}, "yaw": {"min": -np.pi, "max": np.pi}, "pitch": {"min": -np.pi, "max": np.pi}}
        if "show_bounds" not in params:
            params["show_bounds"] = False

        self._params = deepcopy(params)
        
        self.observation_params = observation_params
        
            
        self.observation = None
        # geometry parameters
        self.name = params["location"]["level"]+params["location"]["column"]
        
        self._collisions = {}
        
        self._err_z = {"prev": 0, "accum": 0}
        self._err_th1 = {"prev": 0, "accum": 0}
        self._err_th2 = {"prev": 0, "accum": 0}
        self._err_cam_yaw = {"prev": 0, "accum": 0}
        self._err_cam_pitch = {"prev": 0, "accum": 0}
        
        self.mode = "0" # previously state need to separate out for dm
        
        self.is_right_arm = True
        if params["location"]["column"] == "left":
            self.is_right_arm = False
        
        temp = deepcopy(self._params["pose"]["linear"])
        temp.append(self._params["pose"]["angular"][0])
        
        if "type" not in self.observation_params:
            self.observation_params["type"] = "all"  
        self.observation_params["position"] = [0,0,0]
        pitch = -self._params["pose"]["angular"][0] + np.sum(self._params["pose"]["angular"][1:4])
        self.observation_params["orientation"] = [pitch,self._params["pose"]["angular"][4],0 ]
        if self.observation_params["type"] == "all":
            self.observation = FlowerAll(self.observation_params)
        elif self.observation_params["type"] == "camera":
            self.observation = FlowerPyramid(self.observation_params)
        
        self.update(temp)   
        
               
        
    def update(self, shoulder_pose : list = None, joint_velocity: list = None, constraints : dict = None):
        """
        Update the arm's pose, velocity, and bounding boxes based on movement of supporting body.
        
        :param shoulder_pose: (list) [x,y,z,h] position and angle (rad) of rotation in global frame
        :param joint_velocity: ([v,w1,w2,cam_yaw,cam_pitch]) velocity of the joints.
        :param constraints: (list) {"joints": [z, shoulder, elbow], "bounds": {}} position of the joints.
        """
        self._log.debug("Update Stickbug Arm "+self.name)
        
        if shoulder_pose is None:
            shoulder_pose = deepcopy(self._params["pose"]["linear"])
            shoulder_pose.append(self._params["pose"]["angular"][0])
            
        if joint_velocity is not None:
            self._params["velocity"]["linear"][2] = joint_velocity[0]
            self._params["velocity"]["angular"][1:5] = joint_velocity[1:5]
        
        if constraints is not None:
            self._params["joint_constraints"] = deepcopy(constraints["joints"])
            self._collisions = deepcopy(constraints["bounds"])
        
        joint_pose = self._params["pose"]["angular"]
        joint_pose[0] = shoulder_pose[3]

        self._params["pose"]["linear"][0:2] = deepcopy(shoulder_pose[0:2])
        shoulder, elbow, hand = self.forward_kinematics_absolute()
        self._global_pose = {"shoulder": deepcopy(shoulder)}
        self._global_pose["elbow"] = deepcopy(elbow)
        self._global_pose["hand"] = deepcopy(hand)
        
        self.bounds = {}
        self.bounds[self.name+"_bicep"] = BoundCylinder(self._global_pose["shoulder"][0:3],self._params["buffer"],self._params["mem_length"]["bicep"], -np.pi/2,joint_pose[1]-joint_pose[0],True)
        self.bounds[self.name+"_forearm"] = BoundCylinder(self._global_pose["elbow"][0:3],self._params["buffer"],self._params["mem_length"]["forearm"], -np.pi/2,np.sum(joint_pose[1:3])-joint_pose[0],True)
        
        pitch = -joint_pose[0] + np.sum(joint_pose[1:4])
        self.observation.update_pose(self._global_pose["hand"],[pitch,joint_pose[4],0])

    def forward_kinematics_absolute(self, joints : list = None):
        """
        Calculate the forward kinematics of the arm.
        
        :param joints: ([z, shoulder, elbow]) Joint angles of the arm.
        :return: (list) list containing the absolute pose of each joint.
        """
        
        root = deepcopy(self._params["pose"]["linear"])
        root.append(self._params["pose"]["angular"][0])
        
        
        if joints is None:
            joints = [root[2],self._params["pose"]["angular"][1],self._params["pose"]["angular"][2]]
        
        shoulder, elbow, hand = self.forward_kinematics_relative(joints)
        shoulder = z_rotation_origin(shoulder,root[0:3],-root[3])
        elbow = z_rotation_origin(elbow,root[0:3],-root[3]) #- np.array([0,0,shoulder[2]])
        hand = z_rotation_origin(hand,root[0:3],-root[3]) #- np.array([0,0,elbow[2]])
        #I took negatives off root angle
        #do I need to add local angles on top of thisss??
        return np.array(shoulder), np.array(elbow), np.array(hand)
         
    def forward_kinematics_relative(self,joints):
        """
        Calculate the forward kinematics of the arm.
        
        :param joints: ([z, shoulder, elbow]) Joint angles of the arm.
        :return: (list) list containing the relative pose of each joint.
        """
        z_base,angle_bicep,angle_forearm = joints
        # Calculate positions
        elbow = np.array([self._params["mem_length"]["bicep"],0,0])
        elbow = z_rotation(elbow,[0,0,0],angle_bicep)
        hand = np.array([self._params["mem_length"]["forearm"],0,0])
        hand = z_rotation(hand,[0,0,0],angle_forearm+angle_bicep) + deepcopy(elbow)
        return np.array([0,0,0]), np.array(elbow), np.array(hand)
    
    def plot_arm(self,fig,ax,plot=False):
        """
        Plot the arm.
        
        :param fig: (matplotlib.figure.Figure) Figure to plot on.
        :param ax: (matplotlib.axes._subplots.Axes3DSubplot) Axes to plot on.
        :param plot: (bool) Whether to plot the arm. *default*: False
        """
        ax.plot([self._global_pose["shoulder"][0], self._global_pose["elbow"][0]], [self._global_pose["shoulder"][1], self._global_pose["elbow"][1]], [self._global_pose["shoulder"][2], self._global_pose["elbow"][2]], 'g-', lw=2) # bicep
        ax.plot([self._global_pose["elbow"][0], self._global_pose["hand"][0]], [self._global_pose["elbow"][1], self._global_pose["hand"][1]], [self._global_pose["elbow"][2], self._global_pose["hand"][2]], 'g-', lw=2)  # forearm

        if self._params["show_bounds"]:
            for key in self.bounds:
                self.bounds[key].plot(fig,ax,plot=plot)
        
        # print(self._params["pose"]["angular"])
        # print(np.sum(self._params["pose"]["angular"][0:4]), self._params["pose"]["angular"][4],0)
        self.observation.plot(fig,ax)
        
    def get_bounds(self):
        """
        Get the bounds of the arm.
        
        :return: (dict) Dictionary containing the bounds of the agent.
        """
        return self.bounds
            
    def get_absolute_state(self):
        """
        Get the state of the arm and bounding boxes in global frame.
        
        :return: (dict) Dictionary containing the state of the agent {"arm": {}, "camera": [roll,pitch,yaw], "bounds": {"bicep": {"center": [x,y,z], "radius": r, "height": h, "angle": a}, "forearm": {"center": [x,y,z], "radius": r, "height": h, "angle": a}}
        """
        # print(self._params["pose"]["angular"])
        camera = [0,self._params["pose"]["angular"][4],np.sum(self._params["pose"]["angular"][0:4])]
        p = list(deepcopy(self._global_pose["hand"]))
        p.extend(self._params["pose"]["angular"][1:5])
        v = list(deepcopy(self._params["velocity"]["linear"]))
        v.extend(self._params["velocity"]["angular"])
        return {"position":p, "velocity":v,"bounds": deepcopy(self.bounds)}
    
    def hand_2_position(self,point,dt = 0.1):
        """
        Move the hand to position.
        
        :param point: (list) [x,y,z,th1,th2,cam_yaw,cam_pitch] position of the hand.
        :param dt: (float) time step
        """
        pt = deepcopy(np.array(point[0:3]))
        print(self.name, np.linalg.norm(pt - np.array(self._global_pose["hand"][0:3])))
        pt[2] = self._params["pose"]["linear"][2]
        pt = z_rotation(pt,self._params["pose"]["linear"],self._params["pose"]["angular"][0]-np.pi/2) - self._params["pose"]["linear"]
        pt[2] = point[2]
        angles, _ = arm_2d_ik(pt,self._params["mem_length"]["bicep"],self._params["mem_length"]["forearm"],"L" in self.name,self._params["pose"]["angular"][1:3])
        angles = list(angles)
        angles.extend(point[3:5])
        self.joint_2_position(angles,dt)
        
    def hand_2_relative_position(self,point,dt = 0.1):
        """
        Move the hand to position.
        
        :param point: (list) [x,y,z] position of the hand.
        :param dt: (float) time step
        """
        angles, _ = arm_2d_ik(point,self._params["mem_length"]["bicep"],self._params["mem_length"]["forearm"],"L" in self.name,self._params["pose"]["angular"][1:3])
        angles = list(angles)
        angles.extend(point[3:5])
        self.joint_2_position(angles,dt)

    
    def hand_2_velocity(self,velocity, dt = 0.1):
        """
        Move the hand to velocity. (Need to check if this is accurate.)
        
        :param velocity: (list) [x, y, z, th1,th2,pitch,yaw] velocity of the hand.
        :param dt: (float) time step
        """
        point = self._global_pose["hand"] + np.array(velocity[0:3])*dt
        
        point = np.array(point)
        z = point[2]
        point = z_rotation(point,self._params["pose"]["linear"],self._params["pose"]["angular"][0]-np.pi/2) - self._params["pose"]["linear"]
        point[2] = z
        joint_pose, _ = arm_2d_ik(point,self._params["mem_length"]["bicep"],self._params["mem_length"]["forearm"],"L" in self.name,self._params["pose"]["angular"][1:3])
        joint_pose = list(joint_pose)
        
        dv = np.array(joint_pose) - np.array([self._global_pose["hand"][2], self._params["pose"]["angular"][1], self._params["pose"]["angular"][2]])
        dv /= dt
        # print("DV",dv)
        dv = np.append(dv,velocity[5:7])
        self.joint_2_velocity(dv,dt)
        
            
    def joint_2_position(self,point : list = None, dt = 0.1):
        """
        Move individual joints to position.
        
        :param point: (list) [z, shoulder, elbow, yaw, pitch] position of the joints.
        :param dt: (float) time step
        :return: (dict) Dictionary containing the pose and velocity of the agent.
        """
        self._log.debug("Position Command arm "+self.name+": (z,"+str(point)+"), ( dt, "+str(dt)+")")
        
        err_z = point[0] - self._params["pose"]["linear"][2]
        # print(self.name,"err", err_z, point[0], self._params["pose"]["linear"][2])
        v = self._params["pid"]["p"]*err_z + self._params["pid"]["i"]*self._err_z["accum"] + self._params["pid"]["d"]*(err_z-self._err_z["prev"])/dt
        
        err_th1 = point[1] - self._params["pose"]["angular"][1]
        w1 = self._params["pid_angular"]["p"]*err_th1 + self._params["pid_angular"]["i"]*self._err_th1["accum"] + self._params["pid_angular"]["d"]*(err_th1-self._err_th1["prev"])/dt
        
        err_th2 = point[2] - self._params["pose"]["angular"][2]
        w2 = self._params["pid_angular"]["p"]*err_th2 + self._params["pid_angular"]["i"]*self._err_th2["accum"] + self._params["pid_angular"]["d"]*(err_th2-self._err_th2["prev"])/dt
        
        err_cam_yaw = point[3] - self._params["pose"]["angular"][3]
        w3 = self._params["pid_angular"]["p"]*err_cam_yaw + self._params["pid_angular"]["i"]*self._err_cam_yaw["accum"] + self._params["pid_angular"]["d"]*(err_cam_yaw-self._err_cam_yaw["prev"])/dt
        
        err_cam_pitch = point[4] - self._params["pose"]["angular"][4]
        w4 = self._params["pid_angular"]["p"]*err_cam_pitch + self._params["pid_angular"]["i"]*self._err_cam_pitch["accum"] + self._params["pid_angular"]["d"]*(err_cam_pitch-self._err_cam_pitch["prev"])/dt
        
        self._err_z["accum"] += err_z*dt
        self._err_z["prev"] = err_z
        
        self._err_th1["accum"] += err_th1*dt
        self._err_th1["prev"] = err_th1
        
        self._err_th2["accum"] += err_th2*dt
        self._err_th2["prev"] = err_th2
        
        self._err_cam_yaw["accum"] += err_cam_yaw*dt
        self._err_cam_yaw["prev"] = err_cam_yaw
        
        self._err_cam_pitch["accum"] += err_cam_pitch*dt
        self._err_cam_pitch["prev"] = err_cam_pitch
        
        if abs(err_z) < self._params["pid"]["db"]:
            v = 0
        if abs(err_th1) < self._params["pid_angular"]["db"]:
            w1 = 0
        if abs(err_th2) < self._params["pid_angular"]["db"]:
            w2 = 0
        if abs(err_cam_yaw) < self._params["pid_angular"]["db"]:
            w3 = 0
        if abs(err_cam_pitch) < self._params["pid_angular"]["db"]:
            w4 = 0
        
        self.joint_2_velocity([v,w1,w2,w3,w4],dt)

    def joint_2_velocity(self,velocity : list = None, dt = 0.1):
        """
        Move individual joints to velocity.
        
        :param velocity: (list) [v, w1, w2] velocity of the joints.
        :param dt: (float) time step
        :return: (dict) Dictionary containing the pose and velocity of the agent.
        """
        
        self._log.debug("Velocity Command arm "+self.name+": (v,"+str(velocity)+"), ( dt, "+str(dt)+")")
        # print("velocity: ",velocity)
        if abs(velocity[0]) > self._params["max_speed"]["v"]:
            velocity[0] = np.sign(velocity[0])*self._params["max_speed"]["v"]
        if abs(velocity[1]) > self._params["max_speed"]["w"]:
            velocity[1] = np.sign(velocity[1])*self._params["max_speed"]["w"]
        if abs(velocity[2]) > self._params["max_speed"]["w"]:
            velocity[2] = np.sign(velocity[2])*self._params["max_speed"]["w"]
        if abs(velocity[3]) > self._params["max_speed"]["w"]:
            velocity[3] = np.sign(velocity[3])*self._params["max_speed"]["w"]
        if abs(velocity[4]) > self._params["max_speed"]["w"]:
            velocity[4] = np.sign(velocity[4])*self._params["max_speed"]["w"]
            
        
        z = self._params["pose"]["linear"][2] + velocity[0]*dt  
        # print(self.name,"z",z, velocity[0])    
        # print(self.name,"z",self._params["pose"]["linear"][2],z, self._params["joint_constraints"]["z"]["min"],self._params["joint_constraints"]["z"]["max"])
        for key in self._collisions:
            for p in [self._global_pose["elbow"],self._global_pose["hand"]]:
                if self._collisions[key].contains([p[0],p[1],z]):
                    z = self._params["pose"]["linear"][2]
        z = np.clip(z,self._params["joint_constraints"]["z"]["min"],self._params["joint_constraints"]["z"]["max"])

                
        th1 = self._params["pose"]["angular"][1] + velocity[1]*dt
        for key in self._collisions:
            poses = self.forward_kinematics_absolute([z,th1,self._params["pose"]["angular"][2]])
            for p in [poses[1],poses[2]]:
                if self._collisions[key].contains(p):
                    th1 = self._params["pose"]["angular"][1]
        th1 = np.clip(th1,self._params["joint_constraints"]["th1"]["min"],self._params["joint_constraints"]["th1"]["max"])

        
        th2 = self._params["pose"]["angular"][2] + velocity[2]*dt
        for key in self._collisions:
            poses = self.forward_kinematics_absolute([z,th1,th2])
            for p in [poses[1],poses[2]]:
                if self._collisions[key].contains(p):
                    th2 = self._params["pose"]["angular"][2]
        th2 = np.clip(th2,self._params["joint_constraints"]["th2"]["min"],self._params["joint_constraints"]["th2"]["max"]) 
        
        th3 = self._params["pose"]["angular"][3] + velocity[3]*dt
        th3 = np.clip(th3,self._params["joint_constraints"]["yaw"]["min"],self._params["joint_constraints"]["yaw"]["max"])
        
        th4 = self._params["pose"]["angular"][4] + velocity[4]*dt
        th4 = np.clip(th4,self._params["joint_constraints"]["pitch"]["min"],self._params["joint_constraints"]["pitch"]["max"])

        # print(velocity)
        self._params["pose"]["linear"][2] = z
        self._params["pose"]["angular"][1] = th1
        self._params["pose"]["angular"][2] = th2
        self._params["pose"]["angular"][3] = th3
        self._params["pose"]["angular"][4] = th4
        
        self._params["velocity"]["linear"][2] = velocity[0]
        self._params["velocity"]["angular"][1:5] = velocity[1:5]

        self.update(joint_velocity=velocity)
        
        print(self.name, velocity[1], velocity[2], velocity[3], velocity[4])
        return [self._global_pose["shoulder"], self._global_pose["elbow"], self._global_pose["hand"]]
    
    def step(self, action : dict = None, flowers = None):
        """
        Perform action such as movement or attempt to pollinate
        
        :param action: (dict) Dictionary containing the action to take.
        :param flowers: (list) List of flowers in the environment
        :return: (dict) Dictionary containing observation
        """
        if not(action is None or "command" not in action):
            if "dt" not in action:
                dt = 0.1
            else:
                dt = action["dt"]
            if action["is_joint"]:
                if action["mode"] == "position":
                    self.joint_2_position(action["command"][2:7],dt)
                else:
                    self.joint_2_velocity(action["command"][2:7],dt)
            else:
                if action["mode"] == "position" and not action["is_relative"]:
                    command = action["command"][0:3]
                    command.extend(action["command"][5:7])
                    self.hand_2_position(command,dt)
                elif action["mode"] == "position" and action["is_relative"]:
                    command = action["command"][0:3]
                    command.extend(action["command"][5:7])
                    self.hand_2_relative_position(command,dt)
                else:
                    self.hand_2_velocity(action["command"],dt)
            
        pol_flowers = []
        if "pollinate" in action and action["pollinate"]:
            pol_flowers = self.pollinate_flowers(flowers)
        
        return self.get_absolute_state(), self.observe_flowers(flowers), pol_flowers
    
    def observe_flowers(self,flowers):
        """
        Observe the flowers in the environment
        
        :param flowers: (list) List of flowers in the environment
        :return: (list) List of observations
        """
        pitch = -self._params["pose"]["angular"][0] + np.sum(self._params["pose"]["angular"][1:4])
        return self.observation.observe(flowers, self._global_pose["hand"], [pitch,self._params["pose"]["angular"][4],0 ])

    def pollinate_flowers(self, flowers):
        """
        Pollinate the flowers in the environment
        
        :param flowers: (list) List of flowers in the environment
        :return: (list) Flower that were pollinated
        """
        pos = list(self._global_pose["hand"])
        pos.append(-self._params["pose"]["angular"][0]+np.sum(self._params["pose"]["angular"][1:4]))
        pos.append(self._params["pose"]["angular"][4])
        for flower in flowers:
            if flower.pollinate(pos):
                print("POLLINATED", pos[0:3], flower.position, flower.orientation, flower.is_pollinated)
                #need to make orientation inverse of flower
                return {"position": pos[0:3], "orientation": [0,0,0]}
        return {}
            
    #for pollinate need to pass in and update true flowers