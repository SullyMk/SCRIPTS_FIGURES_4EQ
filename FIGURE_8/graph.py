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

DF, T = read.solution("output/solution_no_fire.txt")
DF0 = DF[0].copy()
DF0.iloc[:,[2,3,4,5]] = DF0.iloc[:,[2,3,4,5]]*0


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
L_Pplus = (np.sum((als.norm_LX(DF0, [MaxDensity, (alpha1/beta1)*MaxDensity, 0, 0]))))

Sum_u1_u2 = np.sqrt(L_x[:,0] + L_x[:, 2])



DF_f, T_f = read.solution("output/solution_with_fire.txt")
DF_f0 = DF_f[0].copy()
DF_f0.iloc[:,[2,3,4,5]] = DF_f0.iloc[:,[2,3,4,5]]*0


lim = np.zeros((len(DF_f)))
L_x_f = np.zeros((len(DF_f), 4))
for i in range(len(DF_f)):
    df = DF_f[i]    
    lim[i]= np.mean(df["u1"]+df["u2"]) - MaxDensity
    L_x_f[i] = als.norm_LX(df)
    
L_x_f1 = (np.sum(L_x_f, axis = 1))


Sum_u1_u2 = (L_x_f[:,0] + L_x_f[:, 2])


Time_to_plot = [T[0], T[int(len(T)/4)], T[int(len(T)/2)], T[int(3*len(T)/4)], T[-1]]
indices_to_plot = [0, int(len(T)/4), int(len(T)/2), int(3*len(T)/4), -1]
Time_to_plot = [int(element) for element in Time_to_plot]

    
grid_x, grid_y = np.meshgrid(
    np.linspace(df['x'].min(), df['x'].max(), 200),
    np.linspace(df['y'].min(), df['y'].max(), 200)
)


# In[]: LOADING FILES WITH FIRE :

Palet = np.arange(0, 1, 0.1).tolist()
        
DF_f = [[] for i in range(len(Palet))]  


for j in Palet:
    
    data, time = read.solution("output/solution_with_firepI"+str(j)+"n"+str(j)+".txt")
    DF_f[Palet.index(j)].append(data)

Norm_evolution = [[[] for n in range(10)] for i in range(len(Palet))]  

if os.path.isfile("output/norm_by_pI.txt"):
    with open("output/norm_by_pI.txt", 'rb') as fichier_pickle:
        Norm_evolution = pic.load(fichier_pickle)
else:
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


# Interpolation avec scipy.griddata
# %%
width_ratios = [1] * (2+len(Time_to_plot)) # Largeur par défaut pour toutes les colonnes sauf la dernière
width_ratios[-1] = 0.1  # Réduire la largeur de la dernière colonne
gs = gridspec.GridSpec(2*(1+len(Palet)), 2+len(Time_to_plot),width_ratios=width_ratios)

N = len(Time_to_plot)

ax = {}  # dictionnaire pour stocker les variables
norm = LogNorm(vmin=1e-2, vmax=MaxDensity)
color_palet =[
    "#00004c",
    "#000074",
    "#00009b",
    "#0000c3",
    "#0000eb",
    "#1116ff",
    "#3244ff",
    "#5471ff",
    "#779eff",
    "#99ccff"
]
# In[]:
fig = plt.figure(figsize=(30, 30))
gs = gridspec.GridSpec(2*(1+len(Palet))+1, 2 + N, width_ratios=[1] + [1]*N + [0.05], wspace=0.3)

ax0 = plt.subplot(gs[:2, 0])
ax0.plot(T, L_x1, color="darkblue")
ax0.set_title('Evolution of U without fire', fontsize=18)
ax0.set(ylabel=r'$\|U(t)\|_X$', ylim=(0, L_Pplus*1.05))
ax0.axhline(y=L_Pplus, color="red", linestyle="--")
ax0.text(
    x=ax0.get_xlim()[0] - 0.025 * (ax0.get_xlim()[1] - ax0.get_xlim()[0]),
    y=L_Pplus*0.99,
    s=r"$\|P^+\|_X$",
    va='center', ha='right', fontsize=18, color='red'
)

for i in range(len(Palet)):
    
    # Créer un sous-graphique et stocker l'objet Axes dans le dictionnaire
    ax[f'ax{i}'] = plt.subplot(gs[3+2*i:3+2*(i+1), 0])  # (nrows, ncols, index)
    if i == 0:
        ax[f'ax{i}'].set_title('Evolution of U with fire', fontsize=18)
    if i < 3 :
        ax[f'ax{i}'].set(ylabel=r'$\|U(t)\|_X$')
        plt.setp(ax[f'ax{i}'].get_xticklabels(), visible=False)
    else:
        ax[f'ax{i}'].set(ylabel=r'$\|U(t)\|_X$', xlabel='time (t)')
    for n in range(1):
        ax[f'ax{i}'].plot(T, Norm_evolution[i][n], color=color_palet[n], linewidth = 0.25, alpha = 0.25)
        

    
        
# === Axes courbes sans feu (haut) et avec feu (bas), partage axe x ===


# === Norme logarithmique partagée pour les heatmaps ===
log_norm = LogNorm(vmin=1e-2, vmax=MaxDensity/2)
ax_list = []
# === Boucle sur les heatmaps ===
for n in range(N):
    df = DF[indices_to_plot[n]]


    grid_u1 = griddata((df['x'], df['y']), df['u1'], (grid_x, grid_y), method='cubic')
    grid_u2 = griddata((df['x'], df['y']), df['u2'], (grid_x, grid_y), method='cubic')
    

    # Plot heatmaps avec imshow
    ax_u1 = plt.subplot(gs[0, 1 + n])
    ax_u2 = plt.subplot(gs[1, 1 + n])
    
    im_u1 = ax_u1.imshow(np.clip(grid_u1, 1e-2, None), origin='lower',
                         extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                         cmap='RdYlGn', norm=log_norm, aspect='auto')

    ax_u2.imshow(np.clip(grid_u2, 1e-2, None), origin='lower',
                 extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                 cmap='RdYlGn', norm=log_norm, aspect='auto')
    
    ax_u1.set_title(r'$u_1(x,y)$', fontweight='bold', fontsize=22)
    ax_u1.set_xticks([])
    ax_u1.set_yticks([])
    
    ax_u2.set_title(r'$u_2(x,y)$', fontweight='bold', fontsize=22)
    ax_u2.set_xticks([])
    ax_u2.set_yticks([])
    
    if n == 0:
        ax_u1.set_ylabel('y')
        ax_u2.set_ylabel('y')
    
    for i in range(len(Palet)):
        df2 = DF_f[i][0][indices_to_plot[n]]
    
    
        grid_u1_f = griddata((df2['x'], df2['y']), df2['u1'], (grid_x, grid_y), method='cubic')
        grid_u2_f = griddata((df2['x'], df2['y']), df2['u2'], (grid_x, grid_y), method='cubic')
        
        ax[f'axh{i}'] = plt.subplot(gs[3+2*i, 1 + n])
        ax[f'axh{i}'].imshow(np.clip(grid_u1_f, 1e-2, None), origin='lower',
                       extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                       cmap='RdYlGn', norm=log_norm, aspect='auto')
       
        ax[f'axhf2{i}'] = plt.subplot(gs[3+2*i+1, 1 + n])
        ax[f'axhf2{i}'].imshow(np.clip(grid_u2_f, 1e-2, None), origin='lower',
                       extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                       cmap='RdYlGn', norm=log_norm, aspect='auto')
        
        ax[f'axh{i}'].set_title(r'$u_1(x,y)$', fontweight='bold', fontsize=22)
        ax[f'axh{i}'].set_xticks([])
        ax[f'axh{i}'].set_yticks([])
        
        ax[f'axhf2{i}'].set_title(r'$u_2(x,y)$', fontweight='bold', fontsize=22)
        ax[f'axhf2{i}'].set_xticks([])
        ax[f'axhf2{i}'].set_yticks([])
        if n == 0:
            ax[f'axh{i}'].set_ylabel('y')
            ax[f'axhf2{i}'].set_ylabel('y')
        if i == 3:
            ax[f'axhf2{i}'].set_xlabel('x')
            ax[f'axhf2{i}'].set_xlabel('x')


# === Titres globaux de colonnes ===
for n in range(N):
    for row in [0, 3]:  # 1ère et 3ème ligne
        ax_col = plt.subplot(gs[row, 1 + n])
        pos = ax_col.get_position()
        fig.text(
            x=pos.x0 + pos.width / 2,
            y=pos.y1 + 0.025,
            s=f"t = {Time_to_plot[n]}",
            ha='center', va='bottom',
            fontsize=18, fontweight='bold'
        )

# === Titres globaux de lignes ===
pos_first_row = plt.subplot(gs[0, 1]).get_position()
pos_last_col = plt.subplot(gs[0, N]).get_position()
x_center = (pos_first_row.x0 + pos_last_col.x0 + pos_last_col.width) / 2

fig.text(
    x=x_center,
    y=pos_first_row.y1 + 0.05,
    s="Simulation without fire",
    ha='center', va='bottom',
    fontsize=30, fontweight='bold'
)

pos_third_row = plt.subplot(gs[3, 1]).get_position()
fig.text(
    x=x_center,
    y=pos_third_row.y1 + 0.05,
    s="Simulation with fire",
    ha='center', va='bottom',
    fontsize=30, fontweight='bold'
)

# === Barre de couleur (colorbar) à droite ===
# Dummy image pour la colorbar si nécessaire
ax_dummy = fig.add_subplot(111, frameon=False)
ax_dummy.set_visible(False)

A = np.linspace(1e-2, MaxDensity/2, 100).reshape(100, 1)
im_dummy = ax_dummy.imshow(A, aspect='auto', cmap='RdYlGn', norm=log_norm)

cbar_ax = plt.subplot(gs[:, 1 + N])
fig.colorbar(im_dummy, cax=cbar_ax)

# === Export en PDF haute qualité ===
fig.savefig('output/norm_evolution.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.0)


n_curves = len(Norm_evolution)
cmap = plt.cm.get_cmap("RdYlGn_r", n_curves)

plt.rcParams['text.usetex'] = True
Dplus = (np.sum((als.norm_LX(DF0, [D1plus, (alpha1/beta1)*D1plus, 0, (alpha2/beta2)*0]))))
L_Pplus = (np.sum((als.norm_LX(DF0, [Pplus1, (alpha1/beta1)*Pplus1, Pplus2, (alpha2/beta2)*Pplus2]))))

# In[]:
fig2, ax = plt.subplots()
n = 0 
for i in range(len(Norm_evolution)):
    color = cmap(i)  # on pioche la i-ème couleur dans la palette
    ax.plot(T, Norm_evolution[i][0], color=color, linewidth=0.8, alpha=0.9,
            label= r'$p = I = '+str(round(Palet[i],2))+'$')
ax.axhline(y=L_Pplus, color="darkblue", linestyle=":", linewidth=0.8)
ax.text(
    x=ax0.get_xlim()[0] - 0 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
    y=L_Pplus*0.99,
    s=r"$\|P^+\|_X$",
    va='center', ha='right', fontsize=11, color='darkblue'
)
ax.set(ylabel=r'$\|U(t)\|_X$')
ax.set(xlabel=r'$t$')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig2.savefig('output/norm_evolution_by_p.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.0)

 # In[]
titles = [r"$p = I = "+str(round(i,2))+"$" for i in Palet]
from matplotlib.colors import LinearSegmentedColormap
colors = [(0.6470588235294118, 0.0, 0.14901960784313725), 
          (0.8431372549019608, 0.18823529411764706, 0.15294117647058825), 
          (0.9568627450980393, 0.42745098039215684, 0.2627450980392157), 
          (0.9921568627450981, 0.6823529411764706, 0.3803921568627451), 
          (0.996078431372549, 0.8784313725490196, 0.5450980392156862), 
          (1.0, 1.0, 0.7490196078431373),
          (0.8509803921568627, 0.9372549019607843, 0.5450980392156862), 
          (0.6509803921568628, 0.8509803921568627, 0.41568627450980394), 
          (0.4, 0.7411764705882353, 0.38823529411764707)]
# Créer une colormap personnalisée
custom_cmap = LinearSegmentedColormap.from_list('rd_yl_cn', colors, N=256)
 
N = len(Palet)
fig3 = plt.figure(figsize=(30, 25))
gs = gridspec.GridSpec(6, 1 + int(N/2), width_ratios=[1]*int(N/2) + [0.05], wspace=0.3,
                       height_ratios=[0.1, 1, 1, 0.1, 1, 1])


# === Boucle sur les heatmaps ===
for n in range(N):
    if n < 5:
        j = 1
        k = n
    else:
        j = 4
        k = n - 5 
    df = DF_f[n][-1][-1]


    grid_u1 = griddata((df['x'], df['y']), df['u1'], (grid_x, grid_y), method='cubic')
    grid_u2 = griddata((df['x'], df['y']), df['u2'], (grid_x, grid_y), method='cubic')
    

    # Plot heatmaps avec imshow
    ax_u1 = plt.subplot(gs[j, k])
    ax_u2 = plt.subplot(gs[j+1, k])
    
    im_u1 = ax_u1.imshow(np.clip(grid_u1, 1e-2, None), origin='lower',
                         extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                         cmap='RdYlGn', norm=log_norm, aspect='auto')

    ax_u2.imshow(np.clip(grid_u2, 1e-2, None), origin='lower',
                 extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                 cmap=custom_cmap, norm=log_norm, aspect='auto')
    
    ax_u1.set_title(r'$u_1(x,y)$', fontweight='bold', fontsize=30)
    ax_u1.set_xticks([])
    ax_u1.set_yticks([])
    
    ax_u2.set_title(r'$u_2(x,y)$', fontweight='bold', fontsize=30)
    ax_u2.set_xticks([])
    ax_u2.set_yticks([])
    



# Ligne 0 : surtitres au-dessus de la première série (lignes j=1,2)
for k, title in enumerate(titles[:5]):
    ax_text = plt.subplot(gs[0, k])  # 0 → ligne 0, k → colonne
    ax_text.text(0.5, 0.5, title,
                 ha='center', va='center',
                 fontsize=26, fontweight='bold')
    ax_text.axis('off')

# Ligne 3 : surtitres au-dessus de la deuxième série (lignes j=4,5)
for k, title in enumerate(titles[5:]):
    ax_text = plt.subplot(gs[3, k])
    ax_text.text(0.5, 0.5, title,
                 ha='center', va='center',
                 fontsize=26, fontweight='bold')
    ax_text.axis('off')
        
ax_dummy = fig.add_subplot(111, frameon=False)
ax_dummy.set_visible(False)

A = np.linspace(1e-2, Pplus, 5).reshape(5, 1)
im_dummy = ax_dummy.imshow(A, aspect='auto', cmap='RdYlGn')
im_dummy2 = ax_dummy.imshow(A, aspect='auto', cmap=custom_cmap)

vmin, vmax = 0, Pplus

# Exemple de ticks : 0, 1/4, 1/2, 3/4, 1
ticks = [0.05 * vmax,
         0.25 * vmax,
         0.5 * vmax,
         0.75 * vmax,
         vmax]

# Barre de couleur pour u1
cbar_ax0 = plt.subplot(gs[1, -1])
cbar0 = fig.colorbar(im_dummy, cax=cbar_ax0, ticks=ticks)
cbar0.ax.tick_params(labelsize=18)  # Taille des nombres

cbar_ax1 = plt.subplot(gs[2, -1])
cbar1 = fig.colorbar(im_dummy2, cax=cbar_ax1, ticks=ticks)
cbar1.ax.tick_params(labelsize=18)

cbar_ax2 = plt.subplot(gs[4, -1])
cbar0 = fig.colorbar(im_dummy, cax=cbar_ax2, ticks=ticks)
cbar0.ax.tick_params(labelsize=18)  # Taille des nombres

cbar_ax3 = plt.subplot(gs[5, -1])
cbar1 = fig.colorbar(im_dummy2, cax=cbar_ax3, ticks=ticks)
cbar1.ax.tick_params(labelsize=18)

fig3.savefig('output/final_heatmaps_by_p.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.0)
