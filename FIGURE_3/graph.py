# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:06:01 2025

@author: e24h297n
"""

import model
import read
import initialpy as inip
import analysis as als
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import pickle as pic
import os
import re
import pandas as pd
import seaborn as sns
params = {
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
    "tmax": 150,
    "tau": 1,
    "p": 0,
    "I": 0,
    "ree":15,
    "chi1":0.1,
    "chi2":0.1,
    "U0": "input/U0.txt", 
    "U": "output/solution_no_fire.txt",
    "F": "output/fire.txt"
}
filename = params["U"]
for key, value in params.items():
    globals()[key] = value

A = 2*(a1*b1+a2*b2)
rho = a1+ a2 
kappa = alpha1 * delta1 + alpha2 * delta2 + a1*b1**2 + a2*b2**2-c1 - c2



L_1_plus = b1 + np.sqrt((alpha1*delta1-c1)/a1 )
L_2_plus = b2 + np.sqrt((alpha2*delta2-c2)/a2 )

L_1_less = b1 - np.sqrt((alpha1*delta1-c1)/a1 )
L_2_less = b2 - np.sqrt((alpha2*delta2-c2)/a2 )

Pplus1 = (L_1_plus - chi1*(L_2_plus))/(1-chi1*chi2)
Pplus2 = (L_2_plus - chi2*(L_1_plus))/(1-chi1*chi2)

Pless1 = (L_1_less - chi1*(L_2_less))/(1-chi1*chi2)
Pless2 = (L_2_less - chi2*(L_1_less))/(1-chi1*chi2)

D1plus = (L_1_plus - chi1*(L_2_less))/(1-chi1*chi2)
D2plus = (L_2_plus - chi2*(L_1_less))/(1-chi1*chi2)

D1less = (L_1_less - chi1*(L_2_plus))/(1-chi1*chi2)
D2less = (L_2_less - chi2*(L_1_plus))/(1-chi1*chi2)


Pplus = np.max([Pplus1, Pplus2])
MaxDensity = (A + np.sqrt(A**2+4*kappa*rho))/(2*rho) 




inputs = "input"
inputs = os.listdir(inputs)


reading = 1 
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

Ndf = pd.DataFrame(N, columns = [round(0.1*j,2) for j in range(10)], index =  [round(0.1*j,2) for j in range(10)])
Ndf = Ndf.iloc[::-1]

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(Ndf, cmap='RdYlGn', annot=False, ax=ax)
ax.set_xlabel('$I$')
ax.set_ylabel('$p$')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.savefig('output/final_heatmaps_by_p_I.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.0)
