# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:06:56 2024

@author: seand
"""


import numpy as np
import matplotlib.pyplot as plt
import os

def nk_unit_checker(filename):
    with open(filename) as file:
        contents=file.read()
        if 'micron' in contents:
            wavelen, n_dust, k_dust = np.loadtxt(filename, skiprows=7, unpack=True)
        elif '1/cm' in contents:
            wavelen, n_dust, k_dust = np.loadtxt(filename, skiprows=7, unpack=True)
            for i in range(len(wavelen)):
                wavelen[i] = (wavelen[i]**(-1) * 1e5)
    file.close()
    return wavelen, n_dust, k_dust


dust_dir = "C:/UTSA/Research/DUSTY/DUSTY/Lib_nk/"

dustlist = [('oliv_nk_x.nk', 'spheres')]
for j in range(len(dustlist)):
    pathy = os.path.join(dust_dir, dustlist[j][0])
    print(pathy)
    wavelen, n_dust, k_dust = np.loadtxt(pathy, skiprows=7, unpack=True)
    wavelen2, n_dust2, k_dust2 = nk_unit_checker(pathy)
    
'''
fig, ax = plt.subplots()

ax.scatter(lam, cabs, marker='o', label='original')
# ax.scatter(x, terp, marker='o',label='interpolated')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
plt.show()

'''
