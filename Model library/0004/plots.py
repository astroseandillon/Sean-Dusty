# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:45:11 2020

@author: seand
"""
"""
This code loads in 
"""



import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

plt.close('all')

wvlnth1, fTot1 = np.loadtxt('work_with_speck.s001', unpack=True, usecols=(0,1))
wvlnth2, fTot2 = np.loadtxt('work_with_speck.s002', unpack=True, usecols=(0,1))
wvlnth3, fTot3 = np.loadtxt('work_with_speck.s003', unpack=True, usecols=(0,1))




plt.figure()
plt.plot(wvlnth1, fTot1, label='graph 1')
plt.plot(wvlnth2, fTot2, label='graph 2')
plt.plot(wvlnth3, fTot3, label='graph 3')
plt.ylabel('lambda')
plt.xlabel('total frequency')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('my first grad school research plot ')
plt.grid()
plt.show()

