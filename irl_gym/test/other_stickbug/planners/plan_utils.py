"""
This module contains the Utilities for Planning
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

__all__ = ["nearest_point", "random_point", "arm_2d_ik"]

def nearest_point(point, points):
    """
    Get the nearest point to a given point
    
    :param point: (list) [x,y,z]
    :param points: (list) list of [x,y,z] points
    :return: (list) [x,y,z] nearest point
    """
    return points[np.argmin(np.linalg.norm(np.array(points) - np.array(point), axis = 1))]

def random_point(point, points):
    """
    Get a random point from a list of points
    
    :param point: (list) [x,y,z]
    :param points: (list) list of [x,y,z] points
    :return: (list) [x,y,z] random point
    """
    return points[np.random.randint(len(points))]

def arm_2d_ik(point, bicep_length, forearm_length, is_left = False, current_angles = [0,0,0]):
    """
    Calculate the inverse kinematics of the arm.
    
    :param point: (list) [x,y,z] position of the hand.
    :param bicep_length: (float) length of the bicep.
    :param forearm_length: (float) length of the forearm.
    :param is_left: (bool) is the arm on the left side of the body.
    :param current_angles: (list) list of current joint angles.
    :return: (list) list containing the joint angles of the arm.
    """
    
    D = np.sqrt(point[0]**2 + point[1]**2)
    # print(D)
    L1 = bicep_length
    L2 = forearm_length
    elbow = np.pi - np.arccos((L1**2 + L2**2 - D**2) / (2 * L1 * L2))
    # print(elbow)
    if is_left:
        elbow = -elbow
    shoulder = np.arctan2(point[0], point[1]) +  np.arctan2(L2 * np.sin(elbow), L1 + L2 * np.cos(elbow))
    
    #This accounts for axes being upside down
    elbow = -elbow

    is_valid = True
    if D > L1 + L2:
        shoulder = current_angles[0]
        elbow = current_angles[1]
        is_valid = False
    return [point[2],shoulder, elbow], is_valid