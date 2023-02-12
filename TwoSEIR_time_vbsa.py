# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:03:15 2023

@author: zam001
"""
#%% Step 1: (import python modules)

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import SAFEpython.VBSA as VB # module to perform VBSA
import SAFEpython.plot_functions as pf # module to visualize the results
from SAFEpython.model_execution import model_execution # module to execute the model
from SAFEpython.sampling import AAT_sampling, AAT_sampling_extend  # module to
# perform the input sampling
from SAFEpython.util import aggregate_boot # function to aggregate results across bootstrap
# resamples

from TwoSEIR_time import twoseir_test

#%% Step 2: (setup the model)

# Initial values
S0, E10, I10, E20, I20, S10, S20, F10, J10, F20, J20, R0 = 0.998, 0, 0.001, 0, 0.001, 0, 0, 0, 0, 0, 0, 0

# Vector of initial values
y0 = S0, E10, I10, E20, I20, S10, S20, F10, J10, F20, J20, R0

# Timeframe
days = 180
t = np.linspace(0,days,days)

# Parameters: beta1 and beta2 = transmission rates, iotta = incubation time, tau = reduced susceptibility,
# sigma = reduced infectiousness, gamma = recovery rate

M = 7

# Parameter ranges:
xmin = [-4, 0, 0, 0, 0, 0, 0]
xmax = [21, 1, 1, 1, 1, 1, 1]

# Parameter distributions
distr_fun = st.uniform # uniform distribution
# The shape parameters for the uniform distribution are the lower limit and the
# difference between lower and upper limits:
distr_par = [np.nan] * M
for i in range(M):
    distr_par[i] = [xmin[i], xmax[i] - xmin[i]]

# Name of parameters (will be used to customize plots):
X_Labels = ['a','b1', 'b2', 'i', 't', 's', 'g']

# Define output:
#fun_test = HyMod.hymod_nse

#%% Step 3 (Compute first-order and total-order variance-based indices)

# Sample parameter space using the resampling strategy proposed by
# (Saltelli, 2008; for reference and more details, see help of functions
# VBSA.vbsa_resampling and VBSA.vbsa_indices)
samp_strat = 'lhs' # Latin Hypercube
N = 3000 #  Number of samples
X = AAT_sampling(samp_strat, M, distr_fun, distr_par, 2*N)
XA, XB, XC = VB.vbsa_resampling(X)

# Run the model and compute selected model output at sampled parameter
# sets:
YA = model_execution(twoseir_test, XA, t, y0) # shape (N, )
YB = model_execution(twoseir_test, XB, t, y0) # shape (N, )
YC = model_execution(twoseir_test, XC, t, y0) # shape (N*M, )

# Compute main (first-order) and total effects:
Si, STi = VB.vbsa_indices(YA, YB, YC, M)

# Plot results:
plt.figure()
plt.subplot(131)
pf.boxplot1(Si, X_Labels=X_Labels, Y_Label='main effects')
plt.subplot(132)
pf.boxplot1(STi, X_Labels=X_Labels, Y_Label='total effects')
plt.subplot(133)
pf.boxplot1(STi-Si, X_Labels=X_Labels, Y_Label='interactions')
plt.show()

# Plot main and total effects in one plot:
plt.figure()
pf.boxplot2(np.stack((Si, STi)), X_Labels=X_Labels,
            legend=['main effects', 'total effects'])
plt.show()

# Check the model output distribution (if multi-modal or highly skewed, the
# variance-based approach may not be adequate):
Y = np.concatenate((YA, YC))
plt.figure()
pf.plot_cdf(Y, Y_Label='NSE')
plt.show()
plt.figure()
fi, yi = pf.plot_pdf(Y, Y_Label='NSE')
plt.show()

# Use bootstrapping to derive confidence bounds:
Nboot = 1000
# Compute sensitivity indices for Nboot bootstrap resamples:
Si, STi = VB.vbsa_indices(YA, YB, YC, M, Nboot=Nboot)
# Si and STi have shape (Nboot, M)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Si_m, Si_lb, Si_ub = aggregate_boot(Si) # shape (M,)
STi_m, STi_lb, STi_ub = aggregate_boot(STi) # shape (M,)
Inti_m, Inti_lb, Inti_ub = aggregate_boot(STi-Si) # shape (M,)

# Plot bootstrapping results:
plt.figure() # plot main, total and interaction effects separately
plt.subplot(131)
pf.boxplot1(Si_m, S_lb=Si_lb, S_ub=Si_ub, X_Labels=X_Labels, Y_Label='main effects')
plt.subplot(132)
pf.boxplot1(STi_m, S_lb=STi_lb, S_ub=STi_ub, X_Labels=X_Labels, Y_Label='total effects')
plt.subplot(133)
pf.boxplot1(Inti_m, S_lb=Inti_lb, S_ub=Inti_ub, X_Labels=X_Labels, Y_Label='interactions')
plt.show()

# Plot main and total effects in one plot:
plt.figure()
pf.boxplot2(np.stack((Si_m, STi_m)), S_lb=np.stack((Si_lb, STi_lb)),
            S_ub=np.stack((Si_ub, STi_ub)), X_Labels=X_Labels,
            legend=['main effects', 'total effects'])
plt.show()

# Analyze convergence of sensitivity indices:
NN = np.linspace(N/10, N, 10).astype(int)
Sic, STic = VB.vbsa_convergence(YA, YB, YC, M, NN)
# Plot convergence results:
plt.figure()
plt.subplot(121)
pf.plot_convergence(Sic, NN*(M+2), X_Label='no of model evaluations',
                    Y_Label='main effects', labelinput=X_Labels)
plt.subplot(122)
pf.plot_convergence(STic, NN*(M+2), X_Label='no of model evaluations',
                    Y_Label='total effects', labelinput=X_Labels)
plt.show()

# Analyze convergence using bootstrapping to derive confidence intervals:
Sic, STic = VB.vbsa_convergence(YA, YB, YC, M, NN, Nboot)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Sic_m, Sic_lb, Sic_ub = aggregate_boot(Sic) # shape (R,M)
STic_m, STic_lb, STic_ub = aggregate_boot(STic) # shape (R,M)

# Plot convergence results:
plt.figure()
plt.subplot(121)
pf.plot_convergence(Sic_m, NN*(M+2), Sic_lb, Sic_ub,
                    X_Label='no of model evaluations',
                    Y_Label='main effects', labelinput=X_Labels)
plt.subplot(122)
pf.plot_convergence(STic_m, NN*(M+2), STic_lb, STic_ub,
                    X_Label='no of model evaluations',
                    Y_Label='total effects', labelinput=X_Labels)
plt.show()

'''

#%% Step 4: Adding up new samples
Nnew = 4000 # increase of base sample size
# (that means: Nnew*(M+2) new samples that will need to be evaluated)
Xext = AAT_sampling_extend(X, distr_fun, distr_par, 2*(N+Nnew)) # extended sample
# (it includes the already evaluated samples 'X' and the new ones)
Xnew = Xext[2*N:2*(N+Nnew), :] # extract the new input samples that need to be
# evaluated

# Resampling strategy:
[XAnew, XBnew, XCnew] = VB.vbsa_resampling(Xnew)
# Evaluate model against new samples:
YAnew = model_execution(twoseir_test, XAnew, t, y0)
# should have shape (Nnew, 1)
YBnew = model_execution(twoseir_test, XBnew, t, y0)
# should have shape (Nnew, 1)
YCnew = model_execution(twoseir_test, XCnew, t, y0)
# should have shape (Nnew*M, 1)

# Put new and old results toghether:
YA2 = np.concatenate((YA, YAnew))  # should have shape (N+Nnew, 1)
YB2 = np.concatenate((YB, YBnew))  # should have shape (N+Nnew,1)
YC2 = np.concatenate((np.reshape(YC, (M, N)), np.reshape(YCnew, (M, Nnew))),
                     axis=1)# should have size (M, N+Nnew)
YC2 = YC2.flatten() # should have size ((N+Nnew)*M, )

# Recompute indices:
Nboot = 1000
Si2, STi2 = VB.vbsa_indices(YA2, YB2, YC2, M, Nboot)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Si2_m, Si2_lb, Si2_ub = aggregate_boot(Si2) # shape (M,)
STi2_m, STi2_lb, STi2_ub = aggregate_boot(STi2) # shape (M,)

# Plot sensitivity indices calculated with the initial sample and the extended
# sample
plt.figure()
plt.subplot(121)
pf.boxplot2(np.stack((Si_m, STi_m)), S_lb=np.stack((Si_lb, STi_lb)),
            S_ub=np.stack((Si_ub, STi_ub)), X_Labels=X_Labels)
plt.title('%d' % (N*(M+2)) + ' model eval.')
plt.subplot(122)
pf.boxplot2(np.stack((Si2_m, STi2_m)), S_lb=np.stack((Si2_lb, STi2_lb)),
            S_ub=np.stack((Si2_ub, STi2_ub)), X_Labels=X_Labels,
            legend=['main effects', 'total effects'])
plt.title('%d' % ((N+Nnew)*(M+2)) + ' model eval.')
plt.show()

#%% Step 6 (Identification of influential and non-influential inputs adding an
# articial 'dummy' input to the list of the model inputs. The sensitivity
# indices for the dummy parameter estimate the approximation error of the
# sensitivity indices. For reference and more details, see help of the function
# VBSA.vbsa_indices)

# Name of parameters (will be used to customize plots) including the dummy input:
X_Labels_dummy = ['a','b1', 'b2', 'i', 't', 's', 'g', 'dummy']

# Compute main (first-order) and total effects using bootstrapping for the model
# inputs and the dummy input:
j = 0
Nboot = 1000
Si, STi, Sdummy, STdummy = VB.vbsa_indices(YA[:, j], YB[:, j], YC[:, j],
                                           M, Nboot=Nboot, dummy=True)
# STdummy is the sensitivity index (total effect) for the dummy input
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Si_m, Si_lb, Si_ub = aggregate_boot(np.column_stack((Si, Sdummy))) # shape (M+1,)
STi_m, STi_lb, STi_ub = aggregate_boot(np.column_stack((STi, STdummy))) # shape (M+1,)

# Plot bootstrapping results:
plt.figure() # plot main and total separately
plt.subplot(121)
pf.boxplot1(Si_m, S_lb=Si_lb, S_ub=Si_ub, X_Labels=X_Labels_dummy, Y_Label='main effects')
plt.subplot(122)
pf.boxplot1(STi_m, S_lb=STi_lb, S_ub=STi_ub, X_Labels=X_Labels_dummy, Y_Label='total effects')
plt.show()

#  Analyze convergence:
NN = np.linspace(N/10, N, 10).astype(int)
Sic, STic, Sdummyc, STdummyc = VB.vbsa_convergence(YA[:, j], YB[:, j], YC[:, j],
                                                   M, NN, Nboot, dummy=True)
# Compute mean and confidence intervals of the sensitivity indices across the
# bootstrap resamples:
Sic_all = []
STic_all = []
for i in range(len(NN)):
    Sic_all = Sic_all + [np.column_stack((Sic[i], Sdummyc[i]))]
    STic_all = STic_all + [np.column_stack((STic[i], STdummyc[i]))]
Sic_m, Sic_lb, Sic_ub = aggregate_boot(Sic_all) # shape (R,M+1)
STic_m, STic_lb, STic_ub = aggregate_boot(STic_all) # shape (R,M+1)

# Plot convergence results:
plt.figure()
plt.subplot(121)
pf.plot_convergence(Sic_m, NN*(M+2), Sic_lb, Sic_ub, X_Label='no of model evaluations',
                    Y_Label='main effects', labelinput=X_Labels_dummy)
plt.subplot(122)
pf.plot_convergence(STic_m, NN*(M+2), STic_lb, STic_ub, X_Label='no of model evaluations',
                    Y_Label='total effects', labelinput=X_Labels_dummy)
plt.show()
'''
