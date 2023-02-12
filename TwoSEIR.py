# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:39:48 2023

@author: zam001
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Initial values
S0, E10, I10, E20, I20, S10, S20, F10, J10, F20, J20, R0 = 0.998, 0, 0.001, 0, 0.001, 0, 0, 0, 0, 0, 0, 0

# Vector of initial values
y0 = S0, E10, I10, E20, I20, S10, S20, F10, J10, F20, J20, R0

# Timeframe
t = np.linspace(0,180,180)

# Parameters: beta1 and beta2 = transmission rates, tau = reduced susceptibility,
# sigma = reduced infectiousness, gamma = recovery rate
beta1, beta2, iotta, tau, sigma, gamma = 0.32, 0.30, 0.8, 0.6, 0.4, 0.2

# TwoSIR model
def twoseir(y, t, beta1, beta2, iotta, tau, sigma, gamma):
    S, E1, I1, E2, I2, S1, S2, F1, J1, F2, J2, R = y
    
    dS = -(beta1*(I1 + sigma*J1) + beta2*(I2 + sigma*J2))*S
    dE1 = beta1*(I1 + sigma*J1)*S - iotta*E1
    dI1 = iotta*E1 - gamma*I1
    dE2 = beta2*(I2 + sigma*J2)*S - iotta*E2
    dI2 = iotta*E2 - gamma*I2
    dS1 = gamma*I1 - beta2*(I2 + sigma*J2)*tau*S1
    dS2 = gamma*I2 - beta1*(I1 + sigma*J1)*tau*S2
    dF1 = beta1*(I1 + sigma*J1)*tau*S2 - iotta*F1
    dJ1 = iotta*F1 - gamma*J1
    dF2 = beta2*(I2 + sigma*J2)*tau*S1 - iotta*F2
    dJ2 = iotta*F2 - gamma*J2
    dR = gamma*(J1 + J2)
    return dS, dE1, dI1, dE2, dI2, dS1, dS2, dF1, dJ1, dF2, dJ2, dR

def twoseir_test(x, t, y0):
    M = 6 # number of model parameters
    if not isinstance(x, np.ndarray):
        raise ValueError('"x" must be a numpy.array.')
        
    if x.dtype.kind != 'f' and x.dtype.kind != 'i' and x.dtype.kind != 'u':
        raise ValueError('"x" must contain floats or integers.')
    Nx = x.shape
    if len(Nx) != 1 or len(x) != M:
        raise ValueError('"x" must have shape (6, ).')
        
    beta1 = x[0]
    beta2 = x[1]
    iotta = x[2]
    tau = x[3]
    sigma = x[4]
    gamma = x[5]
    
    # print(beta1,beta2,tau,sigma,gamma)
    
    soln = odeint(twoseir, y0, t, args=(beta1,beta2,iotta,tau,sigma,gamma))

    
    solnS1 = np.array(soln[:,6])
    solnMax = np.amax(solnS1)
    index = np.where(solnS1 == solnMax)
    print(index)
    print(solnMax)
    
    solnArray = (np.array(solnMax)) 
    
    
    #solnArray = np.array(soln[len(t)-1,7])
    
    return solnArray
        
# Solution of ODE system
soln = odeint(twoseir, y0, t, args=(beta1,beta2,iotta,tau,sigma,gamma))
plt.plot(t, soln[:,0], 'b', label="S")
#plt.plot(t, soln[:,1], 'm', label="I1")
#plt.plot(t, soln[:,2], 'r', label="I2")
plt.plot(t, soln[:,5], 'g', label="S1")
plt.plot(t, soln[:,6], 'y', label ="S2")
#plt.plot(t, soln[:,5], label ="J1")
#plt.plot(t, soln[:,6], label ="J2")
plt.plot(t, soln[:,11], 'k', label ="R")
plt.legend(loc='best')
plt.xlabel('Days')