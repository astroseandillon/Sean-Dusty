# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:10:07 2020

@author: seand
"""


"""
This code loads in spectral components from model0007, details of which can be
found in the README.txt file
This is a model of a binary star system with 30/70 ratio of grf-DL to acC-Hn
"""



import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

plt.close('all')

wvlnth1, fTot1 = np.loadtxt('sphere-2_model0007.s001', unpack=True, usecols=(0,1))
wvlnth2, fTot2 = np.loadtxt('sphere-2_model0007.s002', unpack=True, usecols=(0,1))
wvlnth3, fTot3 = np.loadtxt('sphere-2_model0007.s003', unpack=True, usecols=(0,1))
wvlnth4, fTot4 = np.loadtxt('sphere-2_model0007.s004', unpack=True, usecols=(0,1))
wvlnth5, fTot5 = np.loadtxt('sphere-2_model0007.s005', unpack=True, usecols=(0,1))







plt.figure()
plt.plot(wvlnth1, fTot1, label='Tau = 0.01')
plt.plot(wvlnth2, fTot2, label='Tau = 0.1')
plt.plot(wvlnth3, fTot3, label='Tau = 1.0')
plt.plot(wvlnth4, fTot4, label='Tau = 10.0')
plt.plot(wvlnth5, fTot5, label='Tau = 100.0')
plt.ylabel('lambda')
plt.xlabel('total frequency')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Model 0007 spectra ')
plt.grid()
plt.show()
