#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:05:08 2020

@author: Ryan
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

class Wire:
    '''
    represents an arbitrary 3D wire geometry
    '''
    def __init__(self, current=1, path=None, discretization_length=0.01):
        '''

        :param current: electrical current in Ampere used for field calculations
        :param path: geometry of the wire specified as path of n 3D (x,y,z) points in a numpy array with dimension n x 3
                     length unit is meter
        :param discretizationLength: lenght of dL after discretization
        '''
        self.current = current
        self.path = path
        self.discretization_length = discretization_length
        self.dpath = self.discretized_path()


    def discretized_path(self):
        '''
        calculate end points of segments of discretized path
        approximate discretization length is given by self.discretization_length
        elements will never be combined
        elements longer that self.dicretization_length will be divided into pieces
        :return: discretized path as m x 3 numpy array
        '''

        try:
            return self.dpath
        except AttributeError:
            pass

        self.dpath = deepcopy(self.path)
        for c in range(len(self.dpath)-2, -1, -1):
            # go backwards through all elements
            # length of element
            element = self.dpath[c+1]-self.dpath[c]
            el_len = np.linalg.norm(element)
            npts = int(np.ceil(el_len / self.discretization_length))  # number of parts that this element should be split up into
            if npts > 1:
                # element too long -> create points between
                # length of new sub elements
                sel = el_len / float(npts)
                for d in range(npts-1, 0, -1):
                    self.dpath = np.insert(self.dpath, c+1, self.dpath[c] + element / el_len * sel * d, axis=0)

        return self.dpath


    def calculate_Field_Wire(self, xObs):
        '''
        

        Parameters
        ----------
        xObs : Nx3 array specifying the N observer positions in meters
            Given a set of N observation points, calculate the field from this
            objects self.path carrying self.current.

        Returns
        -------
        bObs : Nx3 array of fields bx, by, bz in Tesla. 

        '''
        
        # pdb.set_trace()
        mu0 = 4e-7*np.pi
        
        dl = self.dpath[1:,:] - self.dpath[0:-1,:] # 1x3 array, dl in Biotsavart law
        
        # define the center point of the wire segment
        xc = (self.dpath[0:-1,:] + self.dpath[1:,:])/2.
        
        # define r' from biotsavart law
        rp = xObs - xc
        rp3 = (np.linalg.norm(rp, axis=1)**3)
        
        bObs = mu0/(4.*np.pi)*self.current*np.cross(dl,rp)/rp3[:,np.newaxis]
        
        return np.sum(bObs, axis=0)       
        
        
        
    def calculate_Field_Seg(self, xObs, xSeg):
        '''
        Deprecated.

        Parameters
        ----------
        xObs : 1x3 array specifying the observer position in meters
        xSeg : 2x3 array specifying the start and end points of a
                wire segment in meters

        Returns
        -------
        bObs: 1x3 array of the field at xObs in Tesla
            

        '''
        mu0 = 4e-7*np.pi
        
        dl = xSeg[1,:] - xSeg[0,:] # 1x3 array, dl in Biotsavart law
        
        # define the center point of the wire segment
        xc = (xSeg[0,:] + xSeg[1,:])/2.
        
        # define r' from biotsavart law
        rp = xObs - xc
        
        bObs = mu0/(4.*np.pi)*self.current*np.cross(dl,rp)/(np.linalg.norm(rp)**3)
        
        return bObs
    
    def translate(self, xyz):
        '''
        move the wire in space
        :param xyz: 3 component vector that describes translation in x,y and z direction
        '''
        if self.dpath is not None:
            self.dpath += np.array(xyz)

        return self    
    

    @staticmethod
    def EllipticalPath(rx=0.1, ry=0.2, pts=20):
        t = np.linspace(0, 2 * np.pi, pts)
        return np.array([rx * np.sin(t), ry * np.cos(t), 0]).T

    @staticmethod
    def CircularPath(radius=0.1, pts=20):
        return Wire.EllipticalPath(rx=radius, ry=radius, pts=pts)

    
    
    def plot_path(self, discretized=True, show=True, ax=None, plt_style='-r'):

        if ax is None:
            fig = plt.figure(None)
            ax = Axes3D(fig)
            
        if discretized:
            p = self.dpath
        else:
            p = self.path   

        ax.plot(p[:, 0], p[:, 1], p[:, 2], plt_style)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # make all axes the same
        #max_a = np.array((p[:, 0], p[:, 1], p[:, 2])).max()

        #ax.set_xlim3d(min(p[:, 0]), max_a)
        #ax.set_ylim3d(min(p[:, 1]), max_a)
        #ax.set_zlim3d(min(p[:, 2]), max_a)


        if show:
            plt.show()

        return ax
        
        
        
        