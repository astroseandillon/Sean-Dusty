#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:17:26 2024

@author: physics
"""

import os
import numpy as np
import scipy.integrate as spit
import matplotlib.pyplot as plt
import time
dust_dir = ['/home/physics/Research/DUSTY/DUSTY/Lib_nk/', 
            "C:/UTSA/Research/DUSTY/DUSTY/Lib_nk/",
           "C:/Users/uhe082/OneDrive - University of Texas at San Antonio/Lib_nk"]
# this is the possible locations of where dust can be


nk_path = dust_dir[0]               #where the dust is 
def bounds_l1():
    return [0,1]

def bounds_l2(l1):
    return [0,1-l1]


def sigma(m, lamda, v):
    '''
    This is the absorption-scattering coefficient. 
    
    C_abs = sigma * C_sca
    
    sigma = (6pi / k^3 V)*Im(m^2) / abs(m^2 - 1)^2
    
    It is independent of the shape of the particle. 

    Parameters
    ----------
    m : Complex
        Complex index of refraction. m = n + ik
    lamda : list
        wavelength of associated m.
    v : float
        volume of sphere-equivalent particles.

    Returns
    -------
    sig : list
        wavelength-dependent scattering-absoprtion coefficient.

    '''
    sig = []
    for i in range(len(lamda)):
        k = (2.0 * np.pi)/lamda[i]
        term1 = (6.0*np.pi) / (v * (k**3))
        term2 = np.imag((m[i]**2))
        term3 = 1.0 / abs(m[i]**2 - 1)**2
        sig.append(term1 * term2 * term3)
    return sig


def probability(dis_name, l1, l2, lmin=0.05, m1=0, m2=0, d=0):
    '''
    This is the probability distribution as a function of L1 and L2, the 
    geometric parameters. This parameter gets inserted into the integral that 
    calculates the average polarizability per unit volume

    Parameters
    ----------
    dis_name : String
        This specifies the distribution we will be using. 
        'CDE' = Continuous Distribution of Ellipsoids
        'ERCDE' = Externally Restricted CDE, returns CDE if lmin=0
        'tCDE' = truncated CDE, REQUIRES MORE WORK
    l1 : Float
        Largest Geometric Constant, lmin<l1<1.0
    l2 : Float
        Second Largest Geometric Constant, lmin<l2<=l1
    lmin : Float, optional
        Minimum allowed geometric constant.  The default is 0.

    Returns
    -------
    Float
        Function dependent on l1 and l2

    '''
    l3 = 1 - l1 - l2
    if dis_name == 'CDE':
        return 2
    elif dis_name == 'CDE2':
        return 120 * l1 * l2 * l3
    elif dis_name == 'ERCDE':
        return 2/((1 - (3*lmin))**2)
    elif dis_name == 'tCDE':
        return 1/((1-d-m2)*(1-m1-m2-d) - 0.5*((1-d-m2)**2) - m1**2)
    else:
        return True
        
    
print('hello world')

def volume_integrand_mrn(r, q):
    v = r**(-q)
    return v

# UNITS ARE IN MICRONS
rmin = 0.005
rmax = 0.25
q = 3.5

r_integral = spit.quad(volume_integrand_mrn, rmin, rmax, args=q)
r_average = ((1/(rmax - rmin)) * r_integral[0])**(1/-q)



v_avg = (4./3.) * np.pi * r_average**3

# vol = (4./3.) * np.pi * 





def regrid_nk(fname, lam_start, lam_end, datapoints, gridtype):
    '''
    regrids nk files into different wavelength ranges

    Parameters
    ----------
    fname : String - name of the file with dustgrains
    
    lam_start : float - smallest wavelength (in microns)
    
    lam_end : float - largest wavelength (in microns)
    
    datapoints : int - number of data points used 
    
    gridtype : string - What type of grid we want to use

    Returns
    -------
    array with optical constants at given wavelength range

    '''
    f, n, k = np.loadtxt(fname, skiprows=8 , unpack=True )
    if gridtype == 'linear':
        grid = np.linspace(lam_start, lam_end, datapoints)
    elif gridtype == 'log':
        grid = np.geomspace(lam_start, lam_end, datapoints)
    newarr = np.ndarray((datapoints, 3))
    newarr[:,0] = grid
    newarr[:,1] = np.interp(grid, f, n)
    newarr[:,2] = np.interp(grid, f, k)
    ret = open(fname[:-4] + '_reg_{0}_{1}.nk'.format(str(lam_start).replace('.','_'), 
                                                  str(lam_end).replace('.','_')), 'w')
    for b in range(datapoints):
        ret.write(f"{newarr[b,0]} \t {newarr[b,1]} \t {newarr[b,2]} \n ")
    ret.close()
    return newarr

def regrid_title(nk, n1, n2):
    '''
    returns the title of our new nk file that has been regridded

    Parameters
    ----------
    nk : string - title of original nk file with .nk ending
        DESCRIPTION.
    n1 : float - low end of datarange
        DESCRIPTION.
    n2 : float - high end of datarange
        DESCRIPTION.

    Returns
    -------
    string - title of regridded nk dat

    '''
    s1 = str(n1).replace('.','_')
    s2 = str(n2).replace('.','_')
    fin_str = nk[:-4] + '_reg_{0}_{1}.nk'.format(s1,s2)
    return fin_str

def cabs(m, dis_name, bounds_l2, bounds_l1):
    cabs = []
    if dis_name=='spheres':
        for j in range(len(m)):
            cabs.append(np.imag(3*(m[j]**2 - 1)/(m[j]**2 + 2)))
    else:
        for j in range(len(m)):
            def f(l1, l2, n=m[j], dis_name=dis_name):
                b = 1/(n**2 - 1)
                term1 = 1/3 * 1/(b + l1)
                term2 = 1/3 * 1/(b + l2)
                term3 = 1/3 * 1/(b + 1 - l1 - l2)
            # r = np.real((term1 + term2 + term3)*probability(dis_name, l1, l2))
                q = np.imag((term1 + term2 + term3)*probability(dis_name, l1, l2))
                return q
            # return np.real((term1 + term2 + term3)*probability(dis_name, l1, l2)) + np.imag((term1 + term2 + term3)*probability(dis_name, l1, l2))
            cabs.append(spit.nquad(f, [bounds_l2, bounds_l1])[0])
    return cabs




dustlist = [('beta-SiC.nk', 'spheres')
            ]
#names of the dusts we will use. these are the initial names

namelist = [dustlist[j][0][:-3]+dustlist[j][1]+'.dat' for j in range(len(dustlist))]
#names of the output files

### REGRID PARAMETERS
lam_small = 0.02           #microns
lam_big = 500.0             #microns
dpoints = 1000               #number of datapoints
gridscale = 'linear'           #'log' or 'linear'

for i in range(len(dustlist)):
    regrid_nk(nk_path+dustlist[i][0], lam_small, lam_big, dpoints, gridscale)
    

reg_list = [regrid_title(nk_path+dustlist[j][0],lam_small,lam_big) for j in range(len(dustlist))]




weightlist = [1.0]
# do the regridding BEFORE calculating Cabs and csca!!!!!

lam_final = np.geomspace(0.001, 1000, num=1200)
# lam_final=wavelen

print('starting calculations')
aaa = time.time()

for j in range(len(dustlist)):
    pathy = os.path.join(nk_path, reg_list[j]) #pipeline is open
    print('path = ',pathy)
    wavelen, n_dust, k_dust = np.loadtxt(pathy, skiprows=7, unpack=True)
    # wavelen = 1e4/wavenum
    print(wavelen[0], ' ', n_dust[0], ' ', k_dust[0])
    m = np.array([complex(n_dust[i], k_dust[i]) for i in range(len(wavelen))])
    print('m = ',m[0])
    cab = cabs(m, dustlist[j][1], bounds_l2, bounds_l1)
    print('cab ',cab[0])
    Cabs_array = np.array((cab))
    print('cab array ', Cabs_array[0], ' of shape ', Cabs_array.shape)
    Cabs_array *= (2 * np.pi / (wavelen)) * v_avg
    print('cab array 2pi/wavelength', Cabs_array[0])
    sig = np.array((sigma(m, wavelen, v_avg)))
    print('sigma ',sig[0])
    Csca_array = Cabs_array/sig
    print('csca ',Csca_array[0])
    output = np.transpose((wavelen, Cabs_array, Csca_array))
    print('output ',output[0])
    f = open(dustlist[j][0][:-3]+dustlist[j][1]+'.dat', 'w')
    for i in range(len(output)):
        f.write(f"{output[i,0]} \t {output[i,1]} \t {output[i,2]}\n")
    f.close()
    print('done with dust: ', pathy)

# 
# avg_array = np.ndarray((len(lam_final),3))
# avg_array[:,0] = lam_final
# for j in range(len(lam_final)):
#     avg_array[j,1] = np.average(total_array[:,j,1], weights=weightlist)
#     avg_array[j,2] = np.average(total_array[:,j,2], weights=weightlist)

    
titlestring=''
for g in range(len(namelist)):
    titlestring += namelist[g][:3] + str(weightlist[g]).replace('.','')
    
# f = open(titlestring+'.dat','w')
# for i in range(len(lam_final)):
#     f.write(f"{avg_array[i,0]} \t {avg_array[i,1]} \t {avg_array[i,2]}\n")
# f.close()   

bbb = time.time()
print('This took {:.3f} seconds to run'.format(bbb-aaa))





