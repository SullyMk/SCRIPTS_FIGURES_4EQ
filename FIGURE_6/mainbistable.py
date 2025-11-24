# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 15:03:02 2025

@author: e24h297n
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:48:09 2024

@author: sully
"""
import model
import read
import initialpy as inip
import analysis as als
import numpy as np
# In[] Parameters settings and simulations run 
#%%
params = {
    "alpha1": 1,
    "beta1": 0.185,
    "delta1": 1,
    "a1": 1,
    "b1": 4,
    "c1": 0.01,
    "d1": 50,
    "alpha2": 1,
    "beta2": 0.185,
    "delta2": 1,
    "a2": 1,
    "b2": 4,
    "c2": 0.01,
    "d2": 50,
    "tmax": 100,
    "tau": 1,
    "p": 0,
    "I": 0.9,
    "ree":5,
    "khi":0.1,
    "U0": "input/U0.txt", 
    "U": "output/solution.txt",
    "F": "output/fire.txt"
}
filename = params["U"]
for key, value in params.items():
    globals()[key] = value



A = 2*(a1*b1+a2*b2)
rho = a1+ a2 
kappa = alpha1 * delta1 + alpha2 * delta2 + a1*b1**2 + a2*b2**2-c1 - c2


u1_less = b1 - np.sqrt((alpha1*delta1-c1)/a1) 
u2_less = b2 - np.sqrt((alpha2*delta2-c2)/a2)
Persistance = (A + np.sqrt(A**2+4*kappa*rho))/(2*rho) 

Diff = [Persistance, (alpha1/beta1)*Persistance, Persistance, (alpha1/beta1)*Persistance]
u1_0 = u1_less
u2_0 = u2_less
w1_0 = (alpha1/beta1)*u1_0
w2_0 = (alpha2/beta2)*u2_0
### (f * alpha * delta)/(a*b**2+c+f) 
### (f * alpha * delta)/(c+f)

initial_condition = inip.regions_noise
arg = [
       [0, 50, 25, 50, u1_0, u1_0*0.1], 
       [0, 50, 25, 50, w1_0, w1_0*0.1],
       [0, 50, 0, 25, u2_0, u2_0*0.1],
       [0, 50, 0, 25, w2_0, w2_0*0.1]
       ]

U0 = model.gen_CI(inip.regions_noise, arg)
DF, T = model.sim_PDE(params)





# In[] Solution visualisation

Time_to_plot = list(range(0, len(T), 5))

read.clean()

for t in Time_to_plot:
    read.plot2D(DF[t], T[t], "output/heatmaps/plot_at"+str(T[t]))

lim = np.zeros((len(DF)))
for i in range(len(DF)):
    df = DF[i]    
    lim[i]= np.mean(df["u1"]+df["u2"]) - Persistance
