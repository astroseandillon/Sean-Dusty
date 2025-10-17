#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 09:07:22 2025

@author: physics
"""

import numpy as np
import scipy.integrate as spit
import matplotlib.pyplot as plt
dust_dir = ['/home/physics/Research/DUSTY/DUSTY/Lib_nk/', 
            "C:/UTSA/Research/DUSTY/DUSTY/Lib_nk/",
           "C:/Users/uhe082/OneDrive - University of Texas at San Antonio (1)/Lib_nk/"]
# this is the possible locations of where dust can be


nk_path = dust_dir[2]               #where the dust is 

def volume_integrand_mrn(r, q):
    v = r**(-q)
    return v

# UNITS ARE IN MICRONS
rmin = 0.005
rmax = 0.25
q = 3.5

r_integral = spit.quad(volume_integrand_mrn, rmin, rmax, args=q)
r_average = ((1/(rmax - rmin)) * r_integral[0])**(1/-q)


#creating the mie theory calc function

def q_sca(r, m, lam):
    q = (8/3) * ((2*np.pi*r/lam)**4) * np.real(((m**2 - 1)/(m**2 + 2))**2)
    return q

def q_abs(r, m, lam):
    q = (8*np.pi*r/lam) * np.imag((m**2 -1)/(m**2 + 2))
    return q


dname = ['beta-SiC.nk']
lam, dust_n, dust_k = np.loadtxt(nk_path+dname[0], unpack=True)



print('hello world')





