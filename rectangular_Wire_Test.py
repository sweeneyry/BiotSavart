#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:49:00 2020

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
path = np.array([[3., 0.01, 0.], 
                 [0., 0.01, 0.],
                 [0., -0.01, 0.],
                 [3., -0.01, 0.], 
                 [3., 0.01, 0.]])

# instantiate a wire
w = wire.Wire(path=path, current=1e3, discretization_length=0.0001)


fig = plt.figure(figsize=(14,4))
ax = fig.add_subplot(1, 3, 1, projection='3d')
w.plot_path(ax=ax)
ax.set_xlim([-2, 4])
ax.set_ylim([-3,3])
ax.set_zlim([-3,3])


# calculate the field 1 m from the wire
xObs = np.zeros(50) + 1.5 #m
yObs = np.linspace(-0.05, 0.05)
print(yObs)
zObs = xObs*0.
CoordObs = np.vstack((xObs, yObs, zObs)).T
ax.plot(xObs, yObs, zObs, color='blue')

numCoords = np.shape(CoordObs)[0]

bObs = np.zeros_like(CoordObs)

for i in range(0,numCoords):
    thisr = CoordObs[i,:]
    bObs[i,:] = w.calculate_Field_Wire(thisr)

# compare with the analytic formula
mu0 = 4e-7*np.pi
bAn = np.array([np.zeros(50), np.zeros(50), -mu0*w.current/(2.*np.pi*(yObs - 0.01)) + mu0*w.current/(2.*np.pi*(yObs + 0.01)) ]).T

ax1 = fig.add_subplot(1, 3, 2)
ax1.plot(yObs*100., bObs[:,2], color='blue', label='Biot-Savart')
ax1.plot(yObs*100., bAn[:,2], '--', color='red', label='Analytic')
ax1.set_xlabel('$Y$ (cm)')
ax1.set_ylabel('$B_z$ (T)')
plt.legend()

ax1 = fig.add_subplot(1, 3, 3)
ax1.semilogy(yObs*100., np.linalg.norm(bObs - bAn, axis=1)/np.linalg.norm(bAn, axis=1), color='blue')
ax1.set_xlabel('$Y$ (cm)')
ax1.set_ylabel('$\|B_{\mathrm{biot}} - B_{\mathrm{an}} \| / \|B_{\mathrm{an}} \|$')

plt.tight_layout()

# print('The field at xObs is ', bObs, ' T')
# print('The normalized error in the Biot-Savart solver is ', np.linalg.norm(bAn -bObs)/np.linalg.norm(bAn))


# contour plot of end effects
xObs = np.linspace(-0.1, 0.1)
yObs = np.linspace(-0.05, 0.05)
XObs, YObs = np.meshgrid(xObs, yObs)
ZObs = XObs*0.
CoordObs = np.dstack((XObs, YObs, ZObs))
iterCoord = CoordObs.reshape((-1,3))

numCoords = np.shape(iterCoord)[0]

bObs = np.zeros_like(iterCoord)

for i in range(0,numCoords):
    thisr = iterCoord[i,:]
    bObs[i,:] = w.calculate_Field_Wire(thisr)
    
bCont = np.linalg.norm(bObs.reshape((50,50,3)), axis=2)

# remove big values close to the wire
cutoff = 2.

bCont[bCont > cutoff] = np.nan

fig = plt.figure()
plt.axes().set_aspect('equal')
plt.contourf(XObs*100., YObs*100., np.log10(bCont))
plt.colorbar(label='$\log_{10}(\|B\|)$ (T)')
plt.xlabel('X (cm)') 
plt.ylabel('Y (cm)')   
plt.tight_layout()








