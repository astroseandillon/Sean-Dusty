# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:16:28 2024

@author: seand
"""


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')



x1, y1, z1 = np.loadtxt('oliv_nk_zCDE.dat', unpack=True)
x2, y2, z2 = np.loadtxt('oliv_nk_yCDE.dat', unpack=True)
x3, y3, z3 = np.loadtxt('oliv_nk_xCDE.dat', unpack=True)
x4, y4, z4 = np.loadtxt('oli10oli10oli10.dat', unpack=True)
# x5, y5, z5 = np.loadtxt('ism.dat', skiprows=3, unpack=True)

x5, y5 = np.loadtxt('cde1_fab01_fig7_olivine.csv', delimiter=',', skiprows=1, unpack=True)
# x6, y6, z6 = np.loadtxt('ISM_std_jumbalaya.dat', unpack=True)
rho = 3.33e-4 # density in g um**-3
v_avg = 3.227383793642055e-05
k4 = y4/(v_avg * rho) * 1e-8



title = 'Benchmark: CDE Comparison of Olivine'


fig,ax = plt.subplots()
ax.set(xscale='log', yscale='log')# xlim=(100,500), ylim=(1e-9,1e-5))
ax.set_title(title, fontsize=16)
ax.set_xlabel(r'$\lambda (\mu m)$', fontsize=14)
ax.set_ylabel(r'$C_{abs}$', fontsize=14)

ax.plot(x1, y1, label='oliv z CDE')
ax.plot(x2, y2, label='oliv y CDE')
ax.plot(x3, y3, label='oliv x CDE')
ax.plot(x4, k4, label='total array')
ax.plot(x5, y5,'.', label='Fabian 2001')
# ax.plot(x6, y6, label='Dillon 2024')
# ax.plot(x6, k6, label='Dillon 2024 kappa')

ax.legend()




plt.show()
