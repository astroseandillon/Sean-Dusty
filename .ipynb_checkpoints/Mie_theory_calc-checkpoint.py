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


nk_path = dust_dir[1]               #where the dust is 

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
wavelen, n_dust, k_dust = np.loadtxt(nk_path+dname[0], unpack=True)
m = np.array([complex(n_dust[i], k_dust[i]) for i in range(len(wavelen))])

qabs_dust = q_abs(r_average, m, wavelen)
qsca_dust = q_sca(r_average, m, wavelen)

cabs_dust = np.pi * r_average * r_average * qabs_dust
csca_dust = np.pi * r_average * r_average * qsca_dust

output = np.transpose((wavelen, cabs_dust, csca_dust))

f = open(dname[0][:-3]+'_mie.dat', 'w')
for i in range(len(output)):
    f.write(f"{output[i,0]} \t {output[i,1]} \t {output[i,2]}\n")
f.close()



print('hello world')





