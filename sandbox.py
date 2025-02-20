# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:06:56 2024

@author: seand
"""


import numpy as np
import matplotlib.pyplot as plt
import os

plt.close('all')


# def nk_unit_checker(filename):
#     with open(filename) as file:
#         contents=file.read()
#         if 'micron' in contents:
#             wavelen, n_dust, k_dust = np.loadtxt(filename, skiprows=7, unpack=True)
#         elif '1/cm' in contents:
#             wavelen, n_dust, k_dust = np.loadtxt(filename, skiprows=7, unpack=True)
#             for i in range(len(wavelen)):
#                 wavelen[i] = (wavelen[i]**(-1) * 1e5)
#     file.close()
#     return wavelen, n_dust, k_dust


# dust_dir = "C:/UTSA/Research/DUSTY/DUSTY/Lib_nk/"

# dustlist = [('oliv_nk_x.nk', 'spheres')]
# for j in range(len(dustlist)):
#     pathy = os.path.join(dust_dir, dustlist[j][0])
#     print(pathy)
#     wavelen, n_dust, k_dust = np.loadtxt(pathy, skiprows=7, unpack=True)
#     wavelen2, n_dust2, k_dust2 = nk_unit_checker(pathy)
    



def kmh(r, a0):
    q = -3.5
    k = r**q * np.exp(-r/a0)
    return k
    
def mrn(r):
    q = -3.5
    return r**q



x = np.geomspace(0.005, 0.25, 100)




fig, ax = plt.subplots()
ax.plot(x, mrn(x), label='MRN')
ax.plot(x, kmh(x, 0.015), label = 'KMH')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
plt.show()


