# -*- coding: utf-8 -*-
"""

Created on 24/11/2024

@author: Sully Mak, Guillaume Cantin, Benoît Delahaye
Nantes Université - LS2N 


"""
import model
import read
import initialpy as inip
import numpy as np
import copy
import pickle as pic
import analysis as als
import matplotlib.pyplot as plt

SIMU = 1
# In[] Parameters settings and simulations run 
#%%
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
    "a2": 0.06,
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
       [0, 50, 0, 50, u1_0, u1_0*0.1], 
       [0, 50, 0, 50, w1_0, w1_0*0.1],
       [0, 50, 0, 50, u2_0, u2_0*0.1],
       [0, 50, 0, 50, w2_0, w2_0*0.1]
       ]

U0 = model.gen_CI(inip.regions_noise, arg)

if SIMU == 1:
    DF, T = model.sim_PDE(params)


DF, T = read.solution(params["U"])
Palet = np.arange(0, 1, 0.1)


if SIMU == 1:
    for j in Palet:
        params_fire = {
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
        params_i["U"] = f'output/solution_with_firepI'+str(j)+'n'+str(j)+'.txt'
        U0 = model.gen_CI(inip.regions_noise, arg)
        DF, T = model.sim_PDE(params_i)
    
DF0 = DF[0].copy()
DF0.iloc[:,[2,3,4,5]] = DF0.iloc[:,[2,3,4,5]]*0

L_Pplus = (np.sum((als.norm_LX(DF0, [Pplus1, (alpha1/beta1)*Pplus1, Pplus2, (alpha2/beta2)*Pplus2]))))
MaxDensity = L_Pplus
lim = np.zeros((len(DF)))
limw = np.zeros((len(DF)))
L_x = np.zeros((len(DF), 4))
for i in range(len(DF)):
    df = DF[i]    
    lim[i]= np.mean(df["u1"]+df["u2"]) - MaxDensity
    limw[i]= np.mean(df["w1"]+df["w2"]) - (alpha1/beta1)*MaxDensity
    L_x[i] = als.norm_LX(df)
    
L_x1 = (np.sum(L_x, axis = 1))
NormPplus = als.norm_LX(DF0, [-MaxDensity, 0, 0, 0])

Sum_u1_u2 = np.sqrt(L_x[:,0] + L_x[:, 2])


Palet = np.arange(0, 1, 0.1).tolist()
        
DF_f = [[] for i in range(len(Palet))]  


for j in Palet:
    
    data, time = read.solution("output/solution_with_firepI"+str(j)+"n"+str(j)+".txt")
    DF_f[Palet.index(j)].append(data)

Norm_evolution = [[[] for n in range(10)] for i in range(len(Palet))]  


for j in range(len(Palet)):
    for n in range(1):
        dataframe = DF_f[j][n]  
        L_x_n = np.zeros((len(dataframe), 4))
        for i in range(len(dataframe)):
            dfi = dataframe[i]
            L_x_n[i] = als.norm_LX(dfi)
        L_x_n = (np.sum(L_x_n, axis = 1))
        Norm_evolution[j][n] = L_x_n
        
with open("output/norm_by_pI.txt", 'wb') as fic_mes:
        FIC = pic.Pickler(fic_mes)
        FIC.dump(Norm_evolution)
n_curves = len(Norm_evolution)
cmap = plt.cm.get_cmap("RdYlGn_r", n_curves)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
fig2, ax = plt.subplots()
n = 0 
for i in range(len(Norm_evolution)):
    color = cmap(i)  # on pioche la i-ème couleur dans la palette
    ax.plot(T, Norm_evolution[i][0], color=color, linewidth=0.8, alpha=0.9,
            label= r'$p = I = '+str(round(Palet[i],2))+'$')
ax.axhline(y=L_Pplus, color="darkblue", linestyle=":", linewidth=0.8)
ax.text(
    x=ax.get_xlim()[0] - 0 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
    y=L_Pplus*0.99,
    s=r"$\|CE_1\|_X$",
    va='center', ha='right', fontsize=11, color='darkblue'
)
ax.set(ylabel=r'$\| \mathbb{U} (t, U_0)\|_X$')
ax.set(xlabel=r'$t$')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig2.savefig('output/norm_evolution_by_p.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.0)
