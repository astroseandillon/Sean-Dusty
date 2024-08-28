#!/usr/bin/env python

import os
import time
import numpy as np

def tcdeopac,rho,lam,e1,e2,m1,m2,delta,cdeop#outcdeop 
     
    dpi=(1d)*!pi 
     
    eps=complex(e1,e2) 
    shapedist=(1d)/((((1d)-delta-m2)*((1d)-m1-m2-delta))-  ((0.5d)*((((1d)-delta-m2)**(2d))-(m1**(2d))))) 
    formula1=((1d)-delta-m2+((1d)/(eps-(1d))))*  alog(((1d)-delta-m2+((1d)/(eps-(1d))))/(m1+((1d)/(eps-(1d))))) 
    formula2=((1d)-delta-m1+((1d)/(eps-(1d))))*  alog(((1d)-delta-m1+((1d)/(eps-(1d))))/(m2+((1d)/(eps-(1d))))) 
    formula3=((1d)-m1-m2+((1d)/(eps-(1d))))*  alog(((1d)-m1-m2+((1d)/(eps-(1d))))/(delta+((1d)/(eps-(1d))))) 
    cdeop=(((2d)*shapedist*dpi)/((3d)*rho*lam))*Imaginary(formula1+formula2+formula3) 
     
    #outcdeop=dblarr(2,n_elements(lam)) 
    #outcdeop[0,*]=lam 
    #outcdeop[1,*]=cdeop 
     
    return cdeop 
     
