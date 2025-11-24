# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:48:09 2024

@author: sully
"""
import model
import read
import initialpy as inip
import numpy as np
import copy
import pickle as pic
import os
import analysis as als
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import pickle as pic
import os
import re
import pandas as pd
import seaborn as sns
# In[] Parameters settings and simulations run 
#%%

SIMU = 0
params = {
    "alpha1": 0.5,
    "beta1": 0.5,
    "delta1": 0.268,
    "a1": 0.06,
    "b1": 0.247,
    "c1": 0.01,
    "d1": 50,
    "alpha2": 0.5,
    "beta2": 0.5,
    "delta2": 0.268,
    "a2": 0.006,
    "b2": 0.247,
    "c2": 0.01,
    "d2": 50,
    "tmax": 200,
    "tau": 1,
    "p": 0,
    "I": 0,
    "ree":10,
    "chi1":0.1,
    "chi2":0.1,
    "U0": "input/U0.txt", 
    "U": "output/solution_no_fire.txt",
    "F": "output/fire.txt"
}
filename = params["U"]
for key, value in params.items():
    globals()[key] = value

if (chi1*chi2 == 1):

    A = 2*(a1*b1+a2*b2)
    rho = a1+ a2 
    kappa = alpha1 * delta1 + alpha2 * delta2 + a1*b1**2 + a2*b2**2-c1 - c2
    
    Pless1 = b1 - np.sqrt((alpha1*delta1-c1)/a1 )
    Pless2 = b2 - np.sqrt((alpha2*delta2-c2)/a2 )
    
    
    MaxDensity = (A + np.sqrt(A**2+4*kappa*rho))/(2*rho) 
    Diff = [MaxDensity, (alpha1/beta1)*MaxDensity, MaxDensity, (alpha1/beta1)*MaxDensity]
    u1_0 = MaxDensity * 0.01
    u2_0 = MaxDensity * 0.01
    w1_0 = (alpha1/beta1)*u1_0
    w2_0 = (alpha2/beta2)*u2_0
    
else:
    L_1_plus = b1 + np.sqrt((alpha1*delta1-c1)/a1 )
    L_2_plus = b2 + np.sqrt((alpha2*delta2-c2)/a2 )

    L_1_less = b1 - np.sqrt((alpha1*delta1-c1)/a1 )
    L_2_less = b2 - np.sqrt((alpha2*delta2-c2)/a2 )

    Pplus1 = (L_1_plus - chi1*(L_2_plus))/(1-chi1*chi2)
    Pplus2 = (L_2_plus - chi2*(L_1_plus))/(1-chi1*chi2)

    Pless1 = (L_1_less - chi1*(L_2_less))/(1-chi1*chi2)
    Pless2 = (L_2_less - chi2*(L_1_less))/(1-chi1*chi2)
    
    
    u1_0 = Pplus1 * 0.01
    u2_0 = Pplus2 * 0.01
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
if SIMU == 1 :
    DF, T = model.sim_PDE(params)
DF, T = read.solution(params["U"])
Palet = np.arange(0, 1, 0.1)

N = 1


if SIMU == 1:
    for j in Palet:
        
        for n in range(N):
            params_fire = {
                "alpha1": 0.5,
                "beta1": 0.5,
                "delta1": 0.268,
                "a1": 0.006,
                "b1": 0.247,
                "c1": 0.01,
                "d1": 50,
                "alpha2": 0.5,
                "beta2": 0.5,
                "delta2": 0.268,
                "a2": 0.06,
                "b2": 0.247,
                "c2": 0.01,
                "d2": 50,
                "tmax": 200,
                "tau": 1,
                "p": j,
                "I": j,
                "ree":10,
                "chi1":0.1,
                "chi2":0.1,
                "U0": "input/U0.txt", 
                "U": "output/solution_with_fire.txt",
                "F": "output/fire.txt"
            }
            params_i = copy.deepcopy(params_fire)
            params_i["U"] = f'input/simu/solution_with_firepI'+str(j)+'n'+str(n)+'.txt'
            U0 = model.gen_CI(inip.regions_noise, arg)
            DF, T = model.sim_PDE(params_i)



# In[] 



inputs = "input/simu"
inputs = os.listdir(inputs)


reading = 0
if reading == 1 :
    DF = [ read.solution("input/"+inputs[n]) for n in range(len(inputs))]
    with open("output/norm_by_pI.txt", 'wb') as fic_mes:
        FIC = pic.Pickler(fic_mes)
        FIC.dump(DF)
else:
    with open("output/norm_by_pI.txt", 'rb') as fichier_pickle:
        DF = pic.load(fichier_pickle)
L_x_f = np.zeros((len(DF), 4))
for i in range(len(DF)):
    df = DF[i]    
    L_x_f[i] = als.norm_LX(df[0][-1])
    
Final_Norm = np.sum(L_x_f, axis = 1)
N = Final_Norm.reshape(10, 10)
plt.rcParams['text.usetex'] = True
Ndf = pd.DataFrame(N, columns = [r'$'+str(round(0.1*j,2))+'$' for j in range(10)], index =  [r'$'+str(round(0.1*j,2))+'$' for j in range(10)])
Ndf = Ndf.iloc[::-1]

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(Ndf, cmap='RdYlGn', annot=False, ax=ax)
ax.set_xlabel('$I$')
ax.set_ylabel('$p$')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.savefig('output/final_heatmaps_by_p_I.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.0)

