"""
This module contains collision checks for an environment
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

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# from matplotlib.patches import Polygon

from abc import ABC, abstractmethod

from irl_gym.utils.tf import *

__all__ = ['BoundPoly', 'BoundBox3D', 'BoundCylinder', 'BoundSphere', 'BoundHexPrism', 'BoundEverything', 'BoundPyramid', 'BoundCamera', 'BoundCone', 'BoundRectPrism']

class BoundPoly(ABC):
    def __init__(self, *args, **kwargs):
        """
        Abstract class for a bounded polygon
        """
        pass
    
    @abstractmethod
    def plot(self, fig, ax, plot):
        """
        Plot the polygon
        """
        raise 
    
    @abstractmethod
    def contains(self, point):
        """
        Check if a point is contained within the polygon
        """
        raise
    

class BoundRectPrism(BoundPoly):
    def __init__(self, corner, dimensions):
        """
        Create a rectangular prism with a corner and dimensions
        
        :param corner: [x, y, z] corner of the prism
        :param dimensions: [x, y, z] dimensions of the prism
        """
        self.corner = np.array(corner)
        self.dimensions = np.array(dimensions)
        
    def plot(self, fig, ax, plot, color = "b"):
        """
        Plot the rectangular prism
        
        :param fig: (fig) figure to plot on
        :param ax: (axis) axis to plot on
        :param plot: (plt) whether to show the plot
        :param color: (str) color of the prism
        """
        
        rectangle = []
        
        x = [self.corner[0], self.corner[0] + self.dimensions[0]]
        y = [self.corner[1], self.corner[1] + self.dimensions[1]]    
  
        for i in range(2):
            for j in range(2):
                rectangle.append([x[i],y[j],self.corner[2]])
        rectangle[2], rectangle[3] = rectangle[3], rectangle[2]

        ax.add_collection3d(Poly3DCollection([rectangle], color="b", alpha=0.1))
        
        rect_top = deepcopy(rectangle)
        for i in range(4):
            rect_top[i][2] += self.dimensions[2]
        ax.add_collection3d(Poly3DCollection([rect_top], color="b", alpha=0.1))
        
        rectangle.append(rectangle[0])
        rect_top.append(rect_top[0])
        for i in range(4):
            rect = np.array([rectangle[i], rectangle[i+1], rect_top[i+1], rect_top[i]])
            ax.add_collection3d(Poly3DCollection([rect], color=color, alpha=0.1))
            
    def contains(self, point):
        """
        check if a point is contained within the prism (not Implemented)
        
        :param point: (np.array) point to check
        :return: (bool) whether the point is contained
        """
        # return super().contains(point)
        raise NotImplementedError("Not implemented")

class BoundBox3D(BoundPoly):
    def __init__(self, p1, p2, p3, buffer=0.1):
        """
        Create a 3D box with 3 points with buffer
        
        :param p1: (np.array) first point
        :param p2: (np.array) second point
        :param p3: (np.array) third point
        :param buffer: (float) buffer around the box
        """
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)
        self.box_min = np.min([self.p1, self.p2, self.p3], axis=0)
        self.box_max = np.max([self.p1, self.p2, self.p3], axis=0)

        # Add a small buffer to the box
        self.box_min[0] += -buffer
        self.box_max[0] += buffer
        
        self.box_min[1] += -buffer
        self.box_max[1] += buffer

        self.box_min[2] += -buffer
        self.box_max[2] += buffer

    def plot(self, fig, ax,plot):
        """
        Plot the box
        
        :param fig: (fig) figure to plot on
        :param ax: (axis) axis to plot on
        :param plot: (plt) whether to show the plot
        """
        # Define edges of the box
        x = [self.box_min[0], self.box_max[0]]
        y = [self.box_min[1], self.box_max[1]]
        z = [self.box_min[2], self.box_max[2]]

        # Create grid for each face and plot surfaces
        xx, yy = np.meshgrid(x, y)
        ax.plot_surface(xx, yy, np.full_like(xx, z[0]), color="b", alpha=0.1)
        ax.plot_surface(xx, yy, np.full_like(xx, z[1]), color="b", alpha=0.1)

        yy, zz = np.meshgrid(y, z)
        ax.plot_surface(np.full_like(yy, x[0]), yy, zz, color="b", alpha=0.1)
        ax.plot_surface(np.full_like(yy, x[1]), yy, zz, color="b", alpha=0.1)

        xx, zz = np.meshgrid(x, z)
        ax.plot_surface(xx, np.full_like(xx, y[0]), zz, color="b", alpha=0.1)
        ax.plot_surface(xx, np.full_like(xx, y[1]), zz, color="b", alpha=0.1)

        # if plot:
        #     plt.show()

    def contains(self, point):
        """
        Check if a point is contained within the box
        
        :param point: (np.array) point to check
        :return: (bool) whether the point is contained
        """
        return all(self.box_min[i] <= point[i] <= self.box_max[i] for i in range(len(point)))

# use x, z rotations to get points on xy plane. Subtract center from both points. Then rotate to consisten plane
class BoundCylinder(BoundPoly):
    """
    Create a cylinder with a center, radius, height, and rotations
    
    :param center: (np.array) center of the cylinder
    :param radius: (float) radius of the cylinder
    :param height: (float) height of the cylinder
    :param y_rot: (float) rotation about the y-axis (rad)
    :param z_rot: (float) rotation about the z-axis (rad)
    :param end_sphere: (bool) whether to add a sphere to the end of the cylinder
    """
    def __init__(self, center, radius, height, y_rot = 0, z_rot = 0, end_sphere = False):
        self.center = np.array(center)
        self.radius = radius
        self.height = height
        self.y_rot = y_rot
        self.z_rot = z_rot
        self.end_sphere = end_sphere

    def plot(self, fig, ax, plot, color="r"):
        """
        Plot the cylinder
        
        :param fig: (fig) figure to plot on
        :param ax: (axis) axis to plot on
        :param plot: (plt) whether to show the plot
        :param color: (str) color of the cylinder
        """
        z = np.linspace(self.center[2], self.center[2] + self.height, 5)
        th = np.linspace(0, 2*np.pi, 50)
        th_grid, z_grid = np.meshgrid(th, z)
        x_grid = self.radius*np.cos(th_grid) + self.center[0]
        y_grid = self.radius*np.sin(th_grid) + self.center[1]
        
        t = np.transpose(np.array([x_grid,y_grid,z_grid]),(1,2,0))
        x_grid, y_grid, z_grid = np.transpose(y_rotation(t,self.center,self.y_rot),(2,0,1))
        t = np.transpose(np.array([x_grid,y_grid,z_grid]),(1,2,0))
        x_grid, y_grid, z_grid = np.transpose(z_rotation(t,self.center,self.z_rot),(2,0,1))
        # if self.y_rot != 0:
        #     for x,y,z in zip(x_grid,y_grid,z_grid):
        #         i= np.shape(x)
        #         for j in range(i):
        #             [x[j],y[j],z[j]] = y_rotation([x[j],y[j],z[j]], self.center, self.y_rot)
            
        # if self.z_rot != 0:
        #     for x,y,z in zip(x_grid,y_grid,z_grid):
        #         i, j = np.shape(x)
        #         for k in range(i):
        #             for l in range(j):
        #                 [x[k,l],y[k,l],z[k,l]] = z_rotation([x[k,l],y[k,l],z[k,l]], self.center, self.z_rot)
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.1)
        
        if self.end_sphere:
            z = deepcopy(self.center)
            z[2] += self.height
            # print(z, self.center, self.y_rot)
            z = y_rotation(z, self.center, self.y_rot)
            # print(z)
            z = z_rotation(z, self.center, self.z_rot)
            # print(z)
            self.sphere = BoundSphere(z, self.radius)
            self.sphere.plot(fig, ax, plot=False)

        # if plot:
        #     plt.show()

    def contains(self, point):
        """
        Check if a point is contained within the cylinder (and end sphere)
        
        :param point: (np.array) point to check
        :return: (bool) whether the point is contained
        """
        if self.end_sphere:
            z = deepcopy(self.center)
            z[2] += self.height
            # print(z, self.center, self.y_rot)
            z = y_rotation(z, self.center, self.y_rot)
            # print(z)
            z = z_rotation(z, self.center, self.z_rot)
            # print(z)
            self.sphere = BoundSphere(z, self.radius)
        
        if self.end_sphere and self.sphere.contains(point):
            return True
        
        start = np.array(self.center)
        end = np.array([start[0],start[1],start[2]+self.height])
        end = y_rotation(end,self.center,self.y_rot)
        end = z_rotation(end,self.center,self.z_rot)
        
        #this hopefully catches if cylinder lies on an axis?
        # diff = end - start
        # if any(diff == 0):
        #     ind = np.where(diff == 0)[0][0]
        #     pt = start[ind]
            
        #     start.pop(ind)
        #     end.pop(ind)
        #     pt2 = point.pop(ind)
            
        #     if abs(pt-pt2) > self.radius:
        #         return False        
        
        dist = np.abs(np.linalg.norm(np.cross(end-start, start-np.array(point))))/np.linalg.norm(end-start)
        if dist < self.radius:
            ang1 = np.arccos(np.dot(end-start, np.array(point)-start)/(np.linalg.norm(end-start)*np.linalg.norm(np.array(point)-start)))
            if ang1 > np.pi/2:
                return False
            ang2 = np.arccos(np.dot(start-end, np.array(point)-end)/(np.linalg.norm(start-end)*np.linalg.norm(np.array(point)-end)))
            if ang2 > np.pi/2:
                return False
            return True
        return False
        
    def collision(self, bound):
        """
        Check if a list of points are contained within the cylinder (and end sphere)
        
        :param points: (list) list of points to check
        :return: (list) list of bools whether the point is contained
        """
        pts = bound.get_points()
        collision = False
        for pt in pts:
            if self.contains(pt):
                collision = True
        return collision
    
    def get_points(self):
        """
        Get the points of the cylinder
        
        :return: (list) list of points
        """
        pt = deepcopy(self.center)
        pt[2] += self.height
        pt = y_rotation(pt,self.center,self.y_rot)
        pt = z_rotation(pt,self.center,self.z_rot)
        pts = np.linspace(self.center,pt,10)
        return pts
        
    
class BoundSphere(BoundPoly):
    def __init__(self, center, radius):
        """
        Create a sphere with a center and radius
        
        :param center: (np.array) center of the sphere
        :param radius: (float) radius of the sphere
        """
        self.center = np.array(center)
        self.radius = radius

    def plot(self, fig, ax, plot):
        """
        Plot the sphere
        
        :param fig: (fig) figure to plot on
        :param ax: (axis) axis to plot on
        :param plot: (plt) whether to show the plot
        """
        # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:5j]
        x = self.radius*np.cos(u)*np.sin(v) + self.center[0]
        y = self.radius*np.sin(u)*np.sin(v) + self.center[1]
        z = self.radius*np.cos(v) + self.center[2]
        ax.plot_surface(x, y, z, color="r", alpha=0.1)

        # if plot:
        #     plt.show()

    def contains(self, point):
        """
        Check if a point is contained within the sphere
        
        :param point: (np.array) point to check
        :return: (bool) whether the point is contained
        """
        point = np.array(point)
        dist = np.sqrt((point[0] - self.center[0])**2 + (point[1] - self.center[1])**2 + (point[2] - self.center[2])**2)
        return dist <= self.radius
    
    def collision(self, bound):
        """
        Check if a list of points are contained within the sphere
        
        :param points: (list) list of points to check
        :return: (list) list of bools whether the point is contained
        """
        pts = bound.get_points()
        collision = False
        for pt in pts:
            if self.contains(pt):
                collision = True
        return collision
    
    def get_points(self):
        """
        Get the points of the sphere
        
        :return: (list) list of points
        """
        return [self.center]
    
    
class BoundHexPrism(BoundPoly):
    def __init__(self, center, radius, height, heading):
        """
        Create a hexagonal prism with a center, radius, height, and heading
        
        :param center: (np.array) center of the hexagonal prism
        :param radius: (float) radius of the hexagonal prism
        :param height: (float) height of the hexagonal prism
        :param heading: (float) heading of the hexagonal prism
        """
        self.update(center=center, radius=radius, height=height, heading=heading)
        
    def update(self, *, center = None, radius = None, height = None, heading= None):
        """
        Update the hexagonal prism parameters
        
        :param center: (np.array) center of the hexagonal prism
        :param radius: (float) radius of the hexagonal prism
        :param height: (float) height of the hexagonal prism
        :param heading: (float) heading of the hexagonal prism
        """
        if center is not None:
            self.center = np.array(center)
        if radius is not None:
            self.radius = radius
        if height is not None:
            self.height = height
        if heading is not None:
            self.heading = heading
        self.get_points()
            
    def get_points(self):
        """
        Get the 2D and 3D corners of the hexagonal prism
        """
        pts_low = []
        pts_high = []
        pts_2d = []
        for i in range(6):
            pts_low.append([self.radius*np.cos((i+1/2)*np.pi/3 + self.heading) + self.center[0], self.radius*np.sin((i+1/2)*np.pi/3 + self.heading) + self.center[1],0])
            pts_high.append([self.radius*np.cos((i+1/2)*np.pi/3 + self.heading) + self.center[0], self.radius*np.sin((i+1/2)*np.pi/3 + self.heading) + self.center[1],self.height])
            pts_2d.append([self.radius*np.cos(i*np.pi/3 + self.heading) + self.center[0], self.radius*np.sin(i*np.pi/3 + self.heading) + self.center[1]])
        self.pts_low = pts_low
        self.pts_high = pts_high
        self.pts_2d = pts_2d
        return pts_2d
        
    def plot(self, fig, ax, plot):
        """
        Plot the hexagonal prism
        
        :param fig: (fig) figure to plot on
        :param ax: (axis) axis to plot on
        :param plot: (plt) whether to show the plot
        """
        ax.add_collection3d(Poly3DCollection([self.pts_low], color="g", alpha=0.1))
        ax.add_collection3d(Poly3DCollection([self.pts_high], color="g", alpha=0.1))
        
        self.pts_low.append(self.pts_low[0])
        self.pts_high.append(self.pts_high[0])
        
        for i in range(6):
            rect = np.array([self.pts_low[i], self.pts_low[i+1], self.pts_high[i+1], self.pts_high[i]])
            ax.add_collection3d(Poly3DCollection([rect], color="g", alpha=0.1))    
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if plot:
            plt.show()
            
    def contains(self, point):
        """
        Check if a point is contained within the hexagonal prism
        
        :param point: (np.array) point to check
        :return: (bool) whether the point is contained
        """
        
        point = Point(point[0],point[1])
        polygon = Polygon(self.pts_2d)
        if not polygon.contains(point):
            return False
        if point[2] < self.center[2] or point[2] > self.center[2] + self.height:
            return False
        return True
        
class BoundEverything(BoundPoly):
    def __init__(self, *args, **kwargs):
        """
        Bound everything
        """
        pass
    
    def plot(self, fig, ax, plot):
        """
        Nothing to plot
        """
        pass 
    
    def contains(self, point):
        """
        Returns true for everything
        """
        return True
    
class BoundPyramid(BoundPoly):
    def __init__(self, center, spread, distance, orientation):
        """
        Bound a pyramid with a center, spread, distance, and orientation
        
        :param center: (np.array) tip of the pyramid
        :param spread: (np.array) spread of the pyramid (x, y)/unit height
        :param distance: (float) distance from the tip to the base
        :param orientation: (np.array) orientation of the pyramid (roll, pitch, yaw)
        """
        # spread rise/run for distance and width
        self.center = np.array(center)
        self.spread = spread # [x, y]
        self.distance = distance
        self.orientation = orientation # [roll pitch yaw]
        self.box = []
        self.triangles = []
        
        self.endpoint = deepcopy(self.center)
        self.endpoint[0] += self.distance
        y = [self.center[1] - self.spread[0]*self.distance, self.center[1] + self.spread[0]*self.distance]
        z = [self.center[2] - self.spread[1]*self.distance, self.center[2] + self.spread[1]*self.distance]
        
        for i in range(2):
            for j in range(2):
                self.box.append([self.endpoint[0],y[i],z[j]])
        self.box[2], self.box[3] = self.box[3], self.box[2]
        
        box_abs = deepcopy(self.box)
        for i in range(len(self.box)):
            self.box[i] = zyx_rotation(self.box[i],self.center,self.orientation) 
        
        self.endpoint = zyx_rotation(self.endpoint,self.center,self.orientation)

        triangle = []
        box_abs.append(box_abs[0])
        for i in range(len(box_abs)-1):
            triangle.clear()
            triangle.append(self.center)
            triangle.append(box_abs[i])
            triangle.append(box_abs[i+1])
            for j in range(1,len(triangle)):
                triangle[j] = zyx_rotation(triangle[j],self.center,self.orientation)
            self.triangles.append(deepcopy(triangle))
    
    def plot(self, fig, ax, plot):
        """
        Plot the pyramid
        
        :param fig: (fig) figure to plot on
        :param ax: (axis) axis to plot on
        :param plot: (plt) whether to show the plot
        """

        # ax.scatter([self.center[0]],[self.center[1]],[self.center[2]],color='blue')
        
        ax.add_collection3d(Poly3DCollection([self.box], color="w", alpha=0.1))  
        # ax.scatter([self.endpoint[0]],[self.endpoint[1]],[self.endpoint[2]],color='blue')

        for triangle in self.triangles:
            ax.add_collection3d(Poly3DCollection([triangle], color="b", alpha=0.05))

    def contains(self, point):
        """
        Check if a point is contained within the pyramid
        
        :param point: (np.array) point to check
        :return: (bool) whether the point is contained
        """
        box = deepcopy(self.box)
        box.append(box[0])
        # print("start")
        s = np.dot(point-self.box[0],np.cross(box[1]-box[0],box[2]-box[0]))
        s = np.sign(s)
        # print(s)
        for i in range(len(self.box)):
            temp = -np.dot(point-self.center,np.cross(box[i]-self.center,box[i+1]-self.center))
            # print(np.sign(temp), s!=0)
            if np.sign(temp) != s and temp != 0 and s != 0:
                return False
        # print("true")
        return True
    
class BoundCamera(BoundPoly):
    def __init__(self, center, spread, radius, orientation):
        pass
    
    def plot(self, fig, ax, plot):
        pass 
    
    def contains(self, point):
        return True
        #chcek if in radius not just same side of normal
        
class BoundCone(BoundPoly):
    def __init__(self, center, spread, distance, orientation):
        pass
    
    def plot(self, fig, ax, plot,color="r"):
        pass 
    
    def contains(self, point):
        return True