# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:44:58 2023

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
days = 180
t = np.linspace(0,days,days)

# Parameters: beta1 and beta2 = transmission rates, tau = reduced susceptibility,
# sigma = reduced infectiousness, gamma = recovery rate
alpha, beta1, beta2, iota, tau, sigma, gamma = 15, 0.5, 0.5, 0.33, 0.5, 0.25, 0.33

#Time when second strain hits

'''def delta(t,alpha):
    if t < alpha:
        return 0
    else:
        return 1'''


def delta(t,alpha):
    return 1/(1+np.exp(alpha - t))


# TwoSIR model
def twoseir(y, t, alpha, beta1, beta2, iota, tau, sigma, gamma):
    S, E1, I1, E2, I2, S1, S2, F1, J1, F2, J2, R = y
    a = delta(t,alpha)
    
    dS = -(beta1*(I1 + sigma*J1) + a*beta2*(I2 + sigma*J2))*S
    dE1 = beta1*(I1 + sigma*J1)*S - iota*E1
    dI1 = iota*E1 - gamma*I1
    dE2 = a*(beta2*(I2 + sigma*J2)*S - iota*E2)
    dI2 = iota*E2 - gamma*I2
    dS1 = gamma*I1 - a*beta2*(I2 + sigma*J2)*tau*S1
    dS2 = gamma*I2 - beta1*(I1 + sigma*J1)*tau*S2
    dF1 = beta1*(I1 + sigma*J1)*tau*S2 - iota*F1
    dJ1 = iota*F1 - gamma*J1
    dF2 = a*beta2*(I2 + sigma*J2)*tau*S1 - iota*F2
    dJ2 = iota*F2 - gamma*J2
    dR = gamma*(J1 + J2)
    return dS, dE1, dI1, dE2, dI2, dS1, dS2, dF1, dJ1, dF2, dJ2, dR

def twoseir_test(x, t, y0):
    M = 7 # number of model parameters
    if not isinstance(x, np.ndarray):
        raise ValueError('"x" must be a numpy.array.')
        
    if x.dtype.kind != 'f' and x.dtype.kind != 'i' and x.dtype.kind != 'u':
        raise ValueError('"x" must contain floats or integers.')
    Nx = x.shape
    if len(Nx) != 1 or len(x) != M:
        raise ValueError('"x" must have shape (7, ).')
       
    alpha = x[0]
    beta1 = x[1]
    beta2 = x[2]
    iota = x[3]
    tau = x[4]
    sigma = x[5]
    gamma = x[6]
    
    # print(beta1,beta2,tau,sigma,gamma)
    
    soln = odeint(twoseir, y0, t, args=(alpha,beta1,beta2,iota,tau,sigma,gamma))

    
    solnS1 = np.array(soln[:,5])
    solnMax = np.amax(solnS1)
    index = np.where(solnS1 == solnMax)
    print(index)
    print(solnMax)
    
    solnArray = (np.array(solnMax)) 
    
    
    #solnArray = np.array(soln[len(t)-1,7])
    
    return solnArray

# Solution of ODE system
soln = odeint(twoseir, y0, t, args=(alpha,beta1,beta2,iota,tau,sigma,gamma))
plt.plot(t, soln[:,0], 'b', label="S")
#plt.plot(t, soln[:,1], 'm', label="I1")
#plt.plot(t, soln[:,2], 'r', label="I2")
plt.plot(t, soln[:,5], 'y', label="S1")
plt.plot(t, soln[:,6], 'g', label ="S2")
#plt.plot(t, soln[:,5], label ="J1")
#plt.plot(t, soln[:,6], label ="J2")
plt.plot(t, soln[:,11], 'k', label ="R")
plt.legend(loc='best')
plt.xlabel('Days')

n = 100

alpha = np.random.normal(14,7,n)
alpha = np.rint(alpha).astype(int)
alpha = np.sort(alpha)
print(alpha)
Susceptible = np.zeros([n,180])
StrainOne = np.zeros([n,180])
StrainTwo = np.zeros([n,180])
BothStrains = np.zeros([n,180])

for i in range(0,n):
    soln = odeint(twoseir, y0, t, args=(alpha[i],beta1,beta2,iota,tau,sigma,gamma))
    Susceptible[i] = soln[:,0]
    StrainOne[i] = soln[:,5]
    StrainTwo[i] = soln[:,6]
    BothStrains[i] = soln[:,11]
    
plt.clf()
axes = plt.axes()
axes.set_ylim([0, 1])
k = np.int(n/2 - 1)
plt.plot(t, StrainTwo[0], '.g', label= "Alpha = " + str(alpha[0]))
plt.plot(t, StrainTwo[k], '--g', label= "Alpha = " + str(alpha[k]))
plt.plot(t, StrainTwo[-1], 'g', label= "Alpha = " + str(alpha[-1]))
plt.plot(t, BothStrains[0], '.k', label= "Alpha = " + str(alpha[0]))
plt.plot(t, BothStrains[k], '--k', label= "Alpha = " + str(alpha[k]))
plt.plot(t, BothStrains[-1], 'k', label= "Alpha = " + str(alpha[-1]))

plt.legend(loc='best')
plt.xlabel('Days')

