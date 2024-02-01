"""
This module contains transforms
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

__all__ = ["x_rotation", "y_rotation", "z_rotation", "z_rotation_origin", "xyz_rotation", "zyx_rotation"]

def x_rotation(point,center,angle):
     """
     Rotate a point about the x axis
     
     :param point: (list) [x,y,z] in global
     :param center: (list) [x,y,z] position of rotation in global frame
     :param angle: (float) angle (rad) of rotation
     :return: (list) [x,y,z] in global frame
     """
     if not isinstance(point, np.ndarray):
          point = np.array(point)
     if not isinstance(center, np.ndarray):
          center = np.array(center)
     R = [[1, 0, 0],
          [0, np.cos(angle), -np.sin(angle)],
          [0, np.sin(angle), np.cos(angle)]]
     return center + (point-center) @ R


def y_rotation(point,center,angle):
     """
     Rotate a point about the y axis
     
     :param point: (list) [x,y,z] in global frame
     :param center: (list) [x,y,z] position of rotation in global frame
     :param angle: (float) angle (rad) of rotation
     """
     if not isinstance(point, np.ndarray):
          point = np.array(point)
     if not isinstance(center, np.ndarray):
          center = np.array(center)
     R = [[np.cos(angle), 0, np.sin(angle)],
         [0, 1,  0],
         [-np.sin(angle),0,np.cos(angle)]]
     return center + (point-center) @ R#,np.array(point)-np.array(center))

def z_rotation(point,center,angle):
     """
     Rotate a point about the z axis
    
     :param point: (list) [x,y,z] in global
     :param center: (list) [x,y,z] position of rotation in global frame
     :param angle: (float) angle (rad) of rotation
     :return: (list) [x,y,z] in global frame
     """
     if not isinstance(point, np.ndarray):
          point = np.array(point)
     if not isinstance(center, np.ndarray):
          center = np.array(center)
     R = [[np.cos(angle), -np.sin(angle), 0],
          [np.sin(angle), np.cos(angle),  0],
          [ 0,0,1]]
     return center + (point -center) @ R

def xyz_rotation(point,center,angles):
     """
     Rotate a point about the x, y, and z axes in that order
     
     :param point: (list) [x,y,z] in global
     :param center: (list) [x,y,z] position of rotation in global frame
     :param angles: (list) [x,y,z] angles (rad) of rotation
     :return: (list) [x,y,z] in global frame
     """
     if not isinstance(point, np.ndarray):
          point = np.array(point)
     if not isinstance(center, np.ndarray):
          center = np.array(center)
     Rx = [[1, 0, 0],
          [0, np.cos(angles[0]), -np.sin(angles[0])],
          [0, np.sin(angles[0]), np.cos(angles[0])]]
     
     Ry = [[np.cos(angles[1]), 0, np.sin(angles[1])],
          [0, 1,  0],
          [-np.sin(angles[1]),0,np.cos(angles[1])]]
     
     Rz = [[np.cos(angles[2]), -np.sin(angles[2]), 0],
          [np.sin(angles[2]), np.cos(angles[2]),  0],
          [0,0,1]]
     return center + (point-center) @ Rz @ Ry @ Rx
           
def zyx_rotation(point,center,angles):
     """
     Rotate a point about the z, y, and x axes in that order
     
     :param point: (list) [x,y,z] in global
     :param center: (list) [x,y,z] position of rotation in global frame
     :param angles: (list) [x,y,z] angles (rad) of rotation 
     :return: (list) [x,y,z] in global frame
     """
     if not isinstance(point, np.ndarray):
          point = np.array(deepcopy(point))
     if not isinstance(center, np.ndarray):
          center = np.array(deepcopy(center))
     Rz = [[np.cos(angles[0]), -np.sin(angles[0]), 0],
           [np.sin(angles[0]), np.cos(angles[0]),  0],
           [0,0,1]]
     Ry = [[np.cos(angles[1]), 0, np.sin(angles[1])],
           [0, 1,  0],
           [-np.sin(angles[1]),0,np.cos(angles[1])]]
     Rx = [[1, 0, 0],
           [0, np.cos(angles[2]), -np.sin(angles[2])],
           [0, np.sin(angles[2]), np.cos(angles[2])]]
     
     return center + (point-center) @ Rx @ Ry @ Rz


def z_rotation_origin(point,center,angle):
     """
     Rotate a point about the z axis
     
     :param point: (list) [x,y,z] in global
     :param center: (list) [x,y,z] position of rotation in global frame
     :param angle: (float) angle (rad) of rotation
     :return: (list) [x,y,z] in global frame
     """
     if not isinstance(point, np.ndarray):
          point = np.array(point)
     if not isinstance(center, np.ndarray):
          center = np.array(center)
     R = [[np.cos(angle), -np.sin(angle), 0],
          [np.sin(angle), np.cos(angle),  0],
          [ 0,0,1]]
     return center + point @ R