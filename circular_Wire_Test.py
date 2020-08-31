#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:46:19 2020

@author: Ryan
"""

import numpy as np
import matplotlib.pyplot as plt
import wire
import matplotlib

fontSize = 12
font = {'family' : 'normal',
'weight' : 'normal',
'size'   : fontSize}
matplotlib.rc('font', **font) 


# current leads to a PF coil

# define the current lead path
npts = 10000
t = np.linspace(0., 2.*np.pi, num=npts)
R = 3. #m
path = np.array([R*np.cos(t), R*np.sin(t), np.zeros(npts)]).T

# instantiate a wire
w = wire.Wire(path=path, current=1e3, discretization_length=0.0005)



fig = plt.figure(figsize=(14,4))
ax = fig.add_subplot(1, 3, 1, projection='3d')
w.plot_path(ax=ax)
# ax.set_xlim([-2, 4])
# ax.set_ylim([-3,3])
# ax.set_zlim([-3,3])


# calculate the field along the axis of the circle
xObs = np.zeros(50) #m
yObs = np.zeros(50)
zObs = np.linspace(-4., 4.)
CoordObs = np.vstack((xObs, yObs, zObs)).T
ax.plot(xObs, yObs, zObs, color='blue')

numCoords = np.shape(CoordObs)[0]

bObs = np.zeros_like(CoordObs)

for i in range(0,numCoords):
    thisr = CoordObs[i,:]
    bObs[i,:] = w.calculate_Field_Wire(thisr)

# compare with the analytic formula
mu0 = 4e-7*np.pi
bzAn = mu0*w.current*R**2/(2.*(R**2 + zObs**2)**(3./2.))
bAn = np.array([np.zeros(50), np.zeros(50), bzAn]).T


ax1 = fig.add_subplot(1, 3, 2)
ax1.plot(zObs, bObs[:,2]*1e4, color='blue', label='Biot-Savart')
ax1.plot(zObs, bAn[:,2]*1e4, '--', color='red', label='Analytic')
ax1.set_xlabel('$Z$ (m)')
ax1.set_ylabel('$B_z$ (G)')
plt.legend()

ax1 = fig.add_subplot(1, 3, 3)
ax1.semilogy(zObs, np.linalg.norm(bObs - bAn, axis=1)/np.linalg.norm(bAn, axis=1), color='blue')
#ax1.semilogy(zObs, np.linalg.norm(bAn, axis=1), color='blue')
ax1.set_xlabel('$Z$ (m)')
ax1.set_ylabel('$\|B_{\mathrm{biot}} - B_{\mathrm{an}} \| / \|B_{\mathrm{an}} \|$')

plt.tight_layout()

