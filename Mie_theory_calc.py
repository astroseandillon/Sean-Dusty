#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 09:07:22 2025

@author: physics
"""

import numpy as np
import matplotlib.pyplot as plt

def q_sca(r, m, lam):
    q = (8/3) * ((2*np.pi*r/lam)**4) * np.real(((m**2 - 1)/(m**2 + 2))**2)
    return q

def q_abs(r, m, lam):
    q = (8*np.pi*r/lam) * np.imag((m**2 -1)/(m**2 + 2))
    return q

print('hello world')