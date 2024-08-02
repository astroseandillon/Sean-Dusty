# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:16:28 2024

@author: seand
"""


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')



x1, y1, z1 = np.loadtxt('ISM_std_CDE.dat', unpack=True)
x2, y2, z2 = np.loadtxt('ISM_std_spheres.dat', unpack=True)
x3, y3, z3 = np.loadtxt('ISM_std_ERCDE.dat', unpack=True)
x4, y4, z4 = np.loadtxt('ISM_std_CDE2.dat', unpack=True)
x5, y5, z5 = np.loadtxt('ism.dat', skiprows=3, unpack=True)





title = 'ISM comparison'


fig,ax = plt.subplots()
ax.set(xscale='log', yscale='log', xlim=(1,60) , ylim=(1e-2, 10))
ax.set_title(title, fontsize=16)
ax.set_xlabel(r'$\lambda (\mu m)$', fontsize=14)
ax.set_ylabel(r'$C_{abs}$', fontsize=14)

ax.plot(x1, y1*10000, label='CDE')
ax.plot(x2, y2*10000, label='Spheres')
ax.plot(x3, y3*10000, label='ERCDE')
ax.plot(x4, y4*10000, label='CDE2')
ax.plot(x5, y5, label='ism_std.dat')


ax.legend()




plt.show()
