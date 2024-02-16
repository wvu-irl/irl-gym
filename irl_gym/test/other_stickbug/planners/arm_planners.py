"""
This module contains the Base Class for Stickbug planning
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

class MultiArmPlanner(ABC):
    """
    Retains multiple arms for planning
    
    **Input**
    
    :param log_level: (int) logging level
    :param planner_type: (str) type of planner to use
    :param arms: (list) list of arms to plan for
    :param arm_params: (dict) dictionary of arm parameters
    """
    def __init__(self, params = None):
        
        super(ArmPlanner, self).__init__()
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        self._log.debug("Init Stickbug Base Planner")
        
        self._base_plan = 
        self._params = deepcopy(params)
    
    def reinit(self, state = None, action = None, s_prime = None):
        """
        Reinitialize Planners
        
        :param state: (dict) dictionary of state
        :param action: (dict) dictionary of action
        :param s_prime: (dict) dictionary of next state
        """
        raise NotImplementedError("reinit not implemented")
        
    @abstractmethod
    def evaluate(self, state):
        """
        Get action from current state
        
        :param state: (dict) dictionary of state
        :return action: (dict) dictionary of action
        
        """
        raise NotImplementedError("evaluate not implemented")

class ArmPlanner(ABC):
    """
    Base class for Stickbug planning
    
    **Input**
    
    :param log_level: (int) logging level
    :param base_params: (dict) dictionary of base parameters
    :param arm_params: (dict) dictionary of arm parameters
    """
    def __init__(self, params = None):
        
        super(ArmPlanner, self).__init__()
        
        if "log_level" not in params:
            params["log_level"] = "WARNING"
        log_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR ,"CRITICAL": logging.CRITICAL}
        ll = log_levels[params["log_level"]]     
                               
        logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=ll)
        self._log = logging.getLogger(__name__)

        self._log.debug("Init Stickbug Base Planner")
        
        self._base_plan = 
        self._params = deepcopy(params)
    
    @abstractmethod
    def reinit(self, state = None, action = None, s_prime = None):
        """
        Reinitialize Planners
        
        :param state: (dict) dictionary of state
        :param action: (dict) dictionary of action
        :param s_prime: (dict) dictionary of next state
        """
        raise NotImplementedError("reinit not implemented")
        
    @abstractmethod
    def evaluate(self, state):
        """
        Get action from current state
        
        :param state: (dict) dictionary of state
        :return action: (dict) dictionary of action
        
        """
        raise NotImplementedError("evaluate not implemented")
    
class GreedyArmPlanner(ArmPlanner):
    
    pass
# sensors
# self.flowers = []
# self.other_arms_current_points = []
# self.other_arms_sides = []

# # memory : Level 0 competancy, avoid arms
# self.current_point = np.array(rest_point)
# self.target_point = np.array(rest_point)
# self.valid_goal = True

# # memory : Level 1 competancy, go to rest position
# self.current_joints = [0,0,0]
# self.rest_joints = [0,0,0]
# self.rest_joints[0],self.rest_joints[1],self.rest_joints[2],valid = np.array(self.inverse_kinematics(rest_point))
# self.current_joints = np.array(self.rest_joints)
# self.target_joints = np.array(self.rest_joints)
# self.rest_point = self.forward_kinematics(self.rest_joints)[2]

# # memory : Level 2 pick random flower
# self.target_flower = np.nan

# # memory: level 3 pick closest flower
# self.time_since_last_pollenation = 0

def greedy_arm_controller(self,ARMS,FLOWERS):

# remove flowers outside of the bounding cylinder
FLOWERS = [flower for flower in FLOWERS if self.Bounding_cylinder.contains(flower)]

# order flowers by closest to current point
distances = np.linalg.norm(np.array(FLOWERS) - np.array(self.current_point), axis=1)
sorted_flowers = np.argsort(distances)
sorted_flowers = np.array(FLOWERS)[sorted_flowers]

# pop the closest flower, check if the point is in collision, if so pop the next closest
while len(sorted_flowers) > 0:
    self.target_point = sorted_flowers[0]
    if not self.check_collision(ARMS, self.target_point):
        break
    sorted_flowers = sorted_flowers[1:]

# if no flowers are able to be reached, go to rest position
if len(sorted_flowers) == 0:
    self.target_point = self.rest_point
    self.target_point[2] = self.current_point[2]
 
# take a small step towards the target point
new_point = 0.1 * (self.target_point - self.current_point) + self.current_point
#calculate the joints
z,shoulder,elbow, self.valid_goal = self.inverse_kinematics(new_point)

if self.valid_goal == True:
    self.current_point = new_point
    self.current_joints = [z, shoulder, elbow]
    shoulder, elbow, hand = self.forward_kinematics(self.current_joints)
    self.bounding_box = BoundBox3D(shoulder, elbow, hand)
    
def arm_controller(self,ARMS,FLOWERS):
# sensory input
self.get_flowers(FLOWERS)
self.get_other_arms(ARMS)

                                                                     
# level 3 competancy : go to closest flower position 
if np.linalg.norm(self.current_point - self.target_point) < .01 and (self.state == "3" or self.state == "2"):
    self.time_since_last_pollenation = 0
else:
    self.time_since_last_pollenation += 1

if self.time_since_last_pollenation < 75 - random.randint(0,25):
    self.target_point = self.pick_closest_flower()
    z,shoulder,elbow, self.valid_goal = self.inverse_kinematics(self.target_point)
    if self.valid_goal == True:
        self.target_joints = [z, shoulder, elbow]
        self.state = "3"
elif self.state == "3":
    self.valid_goal = False


# level 2 competancy : go to a random flower position
if  np.linalg.norm(self.current_point - self.target_point) < .01 or self.valid_goal == False:
    self.target_point = self.pick_random_flower()
    z,shoulder,elbow, self.valid_goal = self.inverse_kinematics(self.target_point)
    if self.valid_goal == True:
        self.target_joints = [z, shoulder, elbow]
        self.state = "2"

 # level 1 competancy : go to the rest position  
if self.valid_goal == False and self.time_since_last_pollenation < 100 - random.randint(0,25):
    self.target_joints = self.rest_joints
    self.state = "1"

# level 0 competancy : avoid other arms in cartesian space
force = self.obav_force(.01*np.linalg.norm(self.current_point - self.target_point))
if np.linalg.norm(force) > .1:
    z,shoulder,elbow, self.valid_goal = self.inverse_kinematics(self.current_point + force)
    if self.valid_goal == True:
        self.target_joints = [z, shoulder, elbow]
        self.state = "0"
        print(force)


# level -1 level competancy : move motors linearly in joint space towards goal angle
shoulder_pt, elbow_pt, self.target_point = self.forward_kinematics(self.target_joints)
self.current_joints  += 0.1 * (self.target_joints - self.current_joints )
shoulder, elbow, hand = self.forward_kinematics(self.current_joints)
self.bounding_box = BoundBox3D(shoulder, elbow, hand)
self.current_point = hand
        

# TODO REDO THE OBAV FORCES
def obav_force(self, scaling_factor=1.0):

repulsive_force = np.zeros(3)

for i in range(len(self.other_arms_current_points)):
    point = self.other_arms_current_points[i]
    other_side = self.other_arms_sides[i]

    vector_to_point = np.array(point) - np.array(self.current_point)
    
    if self.is_right_arm == other_side:
        z_dist = vector_to_point[2] 
        same_side_z_force = [0, 0, -np.sign(z_dist)/z_dist ** 2]
        repulsive_force += same_side_z_force
   

    # Calculate the force magnitude inversely proportional to the distance
    distance = np.linalg.norm(vector_to_point)
    if distance != 0:
        force_magnitude = scaling_factor / (distance ** 2)
        
        # Calculate the force direction
        force_direction = -vector_to_point / distance

        # Accumulate the force
        repulsive_force += force_magnitude * force_direction 
        
return repulsive_force

def pick_random_flower(self):
if len(self.flowers) == 0:
    self.target_flower = np.nan
    return [np.nan,np.nan,np.nan]
self.target_flower = random.randint(0, len(self.flowers) - 1)
return self.flowers[self.target_flower]

def pick_closest_flower(self):
if len(self.flowers) == 0:
    self.target_flower = np.nan
    return [np.nan,np.nan,np.nan]

distances = np.linalg.norm(np.array(self.flowers) - np.array(self.current_point), axis=1)
self.target_flower = np.argmin(distances)

return self.flowers[self.target_flower]

# sensors
# self.flowers = []
# self.other_arms_current_points = []
# self.other_arms_sides = []

# # memory : Level 0 competancy, avoid arms
# self.current_point = np.array(rest_point)
# self.target_point = np.array(rest_point)
# self.valid_goal = True

# # memory : Level 1 competancy, go to rest position
# self.current_joints = [0,0,0]
# self.rest_joints = [0,0,0]
# self.rest_joints[0],self.rest_joints[1],self.rest_joints[2],valid = np.array(self.inverse_kinematics(rest_point))
# self.current_joints = np.array(self.rest_joints)
# self.target_joints = np.array(self.rest_joints)
# self.rest_point = self.forward_kinematics(self.rest_joints)[2]

# # memory : Level 2 pick random flower
# self.target_flower = np.nan

# # memory: level 3 pick closest flower
# self.time_since_last_pollenation = 0

def greedy_arm_controller(self,ARMS,FLOWERS):

# remove flowers outside of the bounding cylinder
FLOWERS = [flower for flower in FLOWERS if self.Bounding_cylinder.contains(flower)]

# order flowers by closest to current point
distances = np.linalg.norm(np.array(FLOWERS) - np.array(self.current_point), axis=1)
sorted_flowers = np.argsort(distances)
sorted_flowers = np.array(FLOWERS)[sorted_flowers]

# pop the closest flower, check if the point is in collision, if so pop the next closest
while len(sorted_flowers) > 0:
    self.target_point = sorted_flowers[0]
    if not self.check_collision(ARMS, self.target_point):
        break
    sorted_flowers = sorted_flowers[1:]

# if no flowers are able to be reached, go to rest position
if len(sorted_flowers) == 0:
    self.target_point = self.rest_point
    self.target_point[2] = self.current_point[2]
 
# take a small step towards the target point
new_point = 0.1 * (self.target_point - self.current_point) + self.current_point
#calculate the joints
z,shoulder,elbow, self.valid_goal = self.inverse_kinematics(new_point)

if self.valid_goal == True:
    self.current_point = new_point
    self.current_joints = [z, shoulder, elbow]
    shoulder, elbow, hand = self.forward_kinematics(self.current_joints)
    self.bounding_box = BoundBox3D(shoulder, elbow, hand)
    
def arm_controller(self,ARMS,FLOWERS):
# sensory input
self.get_flowers(FLOWERS)
self.get_other_arms(ARMS)

                                                                     
# level 3 competancy : go to closest flower position 
if np.linalg.norm(self.current_point - self.target_point) < .01 and (self.state == "3" or self.state == "2"):
    self.time_since_last_pollenation = 0
else:
    self.time_since_last_pollenation += 1

if self.time_since_last_pollenation < 75 - random.randint(0,25):
    self.target_point = self.pick_closest_flower()
    z,shoulder,elbow, self.valid_goal = self.inverse_kinematics(self.target_point)
    if self.valid_goal == True:
        self.target_joints = [z, shoulder, elbow]
        self.state = "3"
elif self.state == "3":
    self.valid_goal = False


# level 2 competancy : go to a random flower position
if  np.linalg.norm(self.current_point - self.target_point) < .01 or self.valid_goal == False:
    self.target_point = self.pick_random_flower()
    z,shoulder,elbow, self.valid_goal = self.inverse_kinematics(self.target_point)
    if self.valid_goal == True:
        self.target_joints = [z, shoulder, elbow]
        self.state = "2"

 # level 1 competancy : go to the rest position  
if self.valid_goal == False and self.time_since_last_pollenation < 100 - random.randint(0,25):
    self.target_joints = self.rest_joints
    self.state = "1"

# level 0 competancy : avoid other arms in cartesian space
force = self.obav_force(.01*np.linalg.norm(self.current_point - self.target_point))
if np.linalg.norm(force) > .1:
    z,shoulder,elbow, self.valid_goal = self.inverse_kinematics(self.current_point + force)
    if self.valid_goal == True:
        self.target_joints = [z, shoulder, elbow]
        self.state = "0"
        print(force)


# level -1 level competancy : move motors linearly in joint space towards goal angle
shoulder_pt, elbow_pt, self.target_point = self.forward_kinematics(self.target_joints)
self.current_joints  += 0.1 * (self.target_joints - self.current_joints )
shoulder, elbow, hand = self.forward_kinematics(self.current_joints)
self.bounding_box = BoundBox3D(shoulder, elbow, hand)
self.current_point = hand
        

# TODO REDO THE OBAV FORCES
def obav_force(self, scaling_factor=1.0):

repulsive_force = np.zeros(3)

for i in range(len(self.other_arms_current_points)):
    point = self.other_arms_current_points[i]
    other_side = self.other_arms_sides[i]

    vector_to_point = np.array(point) - np.array(self.current_point)
    
    if self.is_right_arm == other_side:
        z_dist = vector_to_point[2] 
        same_side_z_force = [0, 0, -np.sign(z_dist)/z_dist ** 2]
        repulsive_force += same_side_z_force
   

    # Calculate the force magnitude inversely proportional to the distance
    distance = np.linalg.norm(vector_to_point)
    if distance != 0:
        force_magnitude = scaling_factor / (distance ** 2)
        
        # Calculate the force direction
        force_direction = -vector_to_point / distance

        # Accumulate the force
        repulsive_force += force_magnitude * force_direction 
        
return repulsive_force

def pick_random_flower(self):
if len(self.flowers) == 0:
    self.target_flower = np.nan
    return [np.nan,np.nan,np.nan]
self.target_flower = random.randint(0, len(self.flowers) - 1)
return self.flowers[self.target_flower]

def pick_closest_flower(self):
if len(self.flowers) == 0:
    self.target_flower = np.nan
    return [np.nan,np.nan,np.nan]

distances = np.linalg.norm(np.array(self.flowers) - np.array(self.current_point), axis=1)
self.target_flower = np.argmin(distances)

return self.flowers[self.target_flower]

