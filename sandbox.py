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
    fin_str = nk[:-3] + '_reg_{0}_{1}.nk'.format(s1,s2)
    return fin_str
    

dustlist = [('sil-dlee.nk', 'spheres'), 
            ('grph1-dl.nk', 'spheres'), 
            ('grph2-dl.nk', 'spheres')]

for i in range(len(dustlist))

# def kmh(r, a0):
#     q = -3.5
#     k = r**q * np.exp(-r/a0)
#     return k
    
# def mrn(r):
#     q = -3.5
#     return r**q



# x = np.geomspace(0.005, 0.25, 100)




# fig, ax = plt.subplots()
# ax.plot(x, mrn(x), label='MRN')
# ax.plot(x, kmh(x, 0.015), label = 'KMH')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.legend()
# plt.show()


