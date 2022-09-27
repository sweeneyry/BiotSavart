#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:24:32 2021

@author: Ryan
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb


def mutual_Inductance(C1, C2):
    
    ''' 
    Calculate the mutual inductance between closed curves 1 and 2
    C1 is an array of xyz coordinates of the closed curve 1. Nx3
    C2 is an array of xyz coordinates of the closed curve 2. Nx3
    '''
    
    if (np.sum(C1[0,:] - C1[-1,:]) >= 1e-10 or np.sum(C2[0,:] - C2[-1,:]) >= 1e-10):
        print('C1 or C2 do not start and end at the same point')
        pdb.set_trace()
    
    # we need to think about line segments rather than points. assuming the input
    # C1 and C2 are arbitrary sets of points, we will find the midpoints of each
    # segment.    
    c1mid = (C1[0:-1,:] + C1[1:,:])/2. 
    c2mid = (C2[0:-1,:] + C2[1:,:])/2. 

    # and now dx, dy, dz
    dc1 = C1[1:,:] - C1[0:-1,:]
    dc2 = C2[1:,:] - C2[0:-1,:]
    
    # the equation we will use is from Wikipedia (https://en.wikipedia.org/wiki/Inductance)
    # we will loop over each midpoint of C1 and integrate over C2
    
    mutual = 0.
    
    for i in range(0, len(c1mid)):
        
        # pull out 'this' midpoint and differential vector
        tc1mid = c1mid[i]
        tdc1 = dc1[i]
        
        dotprod = np.dot(tdc1, dc2.T)
        
        diff = tc1mid - c2mid
        diffmag = np.linalg.norm(diff, axis=1)
        
        mutual += np.sum(dotprod/diffmag)
        
    
        
    # note that the coefficient in front of the integral is mu0/4pi, which is
    # simply 1e-7
    return mutual*1e-7
        

def test_Mutual_Inductance():
    '''
    A test case. Two circular wire loops separated by dz. The upper loop
    is much smaller than the lower loop. This problem approaches an analytic
    solution when the little loop is well approximated by a dipole. 
    http://www.pas.rochester.edu/~dmw/phy217/Lectures/Lect_35b.pdf

    Returns
    -------
    None.

    '''

    r1 = 1. 
    r2 = r1/100.
    phi = np.linspace(0, 2*np.pi, num=1000)
    
    x1 = r1*np.cos(phi)
    y1 = r1*np.sin(phi)
    z1 = phi*0.
    c1 = np.column_stack((x1,y1,z1))

    x2 = r2*np.cos(phi)
    y2 = r2*np.sin(phi)
    z2 = z1 + r1
    c2 = np.column_stack((x2,y2,z2))

    testmut = mutual_Inductance(c1, c2)   
    
    print('Mutual is ', testmut, ' H')
    
    mu0 = 4e-7*np.pi
    anmut = mu0*np.pi/2.*(r1**2*r2**2/(r1**2 + r1**2)**(1.5))
    
    print('The theoretical value is ', anmut, ' H')
    
    
    
