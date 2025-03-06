# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:16:28 2024

@author: seand
"""


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


dust_dir = ['/home/physics/Research/DUSTY/DUSTY/Lib_nk/', 
            "C:/UTSA/Research/DUSTY/DUSTY/Lib_nk/",
           "C:/Users/uhe082/OneDrive - University of Texas at San Antonio/Lib_nk"]
# this is the possible locations of where dust can be


nk_path = dust_dir[1]               #where the dust is 

# x1, y1, z1 = np.loadtxt('ism-stnd.dat', unpack=True,skiprows=3)
# x2, y2, z2 = np.loadtxt('sil530grp3132grp1566.dat', unpack=True)
# x3, y3, z3 = np.loadtxt('oliv_nk_xCDE2.dat', unpack=True)
# y4 = np.average((y1,y2[:-1],y3[:-2]),axis=0)

# x4, y4, z4 = np.loadtxt('oli10oli10oli10.dat', unpack=True)
# x5, y5, z5 = np.loadtxt('ism.dat', skiprows=3, unpack=True)

# x5, y5 = np.loadtxt('cde2_fab01_fig7_olivine_8_13.csv', delimiter=',', skiprows=1, unpack=True)
# x6, y6, z6 = np.loadtxt('ISM_std_jumbalaya.dat', unpack=True)
rho = 3.33e-4 # density in g um**-3
v_avg = 3.227383793642055e-05


title = 'NK regridded'
x1,n1,k1 = np.loadtxt(nk_path+'sil-dle_reg_0_05_1000_0.nk', unpack=True)

x2,n2,k2 = np.loadtxt(nk_path+'sil-dlee.nk',unpack=True,skiprows=8)


# x1=1e4/x1
# x2=1e4/x2
# x3=1e4/x3
# x4=1e4/x4



# k1 = 50*k1/(x1**2)
# k2 = 50*k2/(x2**2)
# k3 = 50*k3/(x3**2)
# k4 = 50*k4/(x4**2)








fig,ax = plt.subplots(2,1)
ax[0].set(xscale='log', yscale='log')
ax[1].set(xscale='log', yscale='log')

ax[0].set_title(title, fontsize=16)
# ax[0].set_xlabel(r'$\lambda (\mu m)$', fontsize=14)
ax[1].set_xlabel(r'Wavelength $(\mu m)$', fontsize=14)
# ax[0].set_ylabel(r'$\kappa$', fontsize=14)
# ax[0].set_xlim(-1,100)
# ax[1].set_xlim(-1,100)

# ax[0].set_ylim(0.0001,100)
# ax[1].set_ylim(0.0001,100)

ax[0].set_ylabel('N')
ax[1].set_ylabel('K')

ax[0].plot(x1,n1,'^g', label='regridded')
ax[0].plot(x2,n2,'r', label='original')

ax[1].plot(x1,k1,'^g', label='regridded')
ax[1].plot(x2,k2,'r', label='original')



# ax.plot(x1, k1, label='oliv z CDE')
# ax.plot(x2, k2, label='oliv y CDE')
# ax.plot(x3, k3, label='oliv x CDE')
# ax[0].plot(x1, y1, label='ISM_std')

# ax[0].plot(x2, y2/v_avg,'.', label='new')
# ax.plot(x6, y6, label='Dillon 2024')
# ax.plot(x6, k6, label='Dillon 2024 kappa')

ax[1].legend(loc='lower right')




plt.show()
