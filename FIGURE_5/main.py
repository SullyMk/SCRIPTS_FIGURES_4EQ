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

params = {
    "alpha1": 1,
    "beta1": 1,
    "delta1": 1 ,
    "a1": 0.5,
    "b1": 3,
    "c1": 0.03,
    "d1": 1,
    "alpha2": 1,
    "beta2": 1,
    "delta2": 1,
    "a2": 0.5,
    "b2": 3,
    "c2": 0.03,
    "d2": 1,
    "tmax": 50,
    "tau": 1,
    "p": 0,
    "I": 0.9,
    "ree":2,
    "chi1":0.05,
    "chi2":0.05,
    "U0": "input/U0.txt", 
    "U": "output/solution_test_with_fire_Pless.txt",
    "F": "output/fire.txt"
}
filename = params["U"]
for key, value in params.items():
    globals()[key] = value

SIMU = 0

A = 2*(a1*b1+a2*b2)
rho = a1+ a2 
kappa = alpha1 * delta1 + alpha2 * delta2 + a1*b1**2 + a2*b2**2 -c1 - c2

L_1_plus = b1 + np.sqrt((alpha1*delta1-c1)/a1 )
L_2_plus = b2 + np.sqrt((alpha2*delta2-c2)/a2 )

L_1_less = b1 - np.sqrt((alpha1*delta1-c1)/a1 )
L_2_less = b2 - np.sqrt((alpha2*delta2-c2)/a2 )

Pplus1 = (L_1_plus - chi1*(L_2_plus))/(1-chi1*chi2)
Pplus2 = (L_2_plus - chi2*(L_1_plus))/(1-chi1*chi2)

Pless1 = (L_1_less - chi1*(L_2_less))/(1-chi1*chi2)
Pless2 = (L_2_less - chi2*(L_1_less))/(1-chi1*chi2)

Pplus = np.max([Pplus1, Pplus2]) 
Diff = [Pplus, (alpha1/beta1)*Pplus, Pplus, (alpha1/beta1)*Pplus]
u1_0 = Pless1 
u2_0 = Pless2
w1_0 = (alpha1/beta1)*u1_0
w2_0 = (alpha2/beta2)*u2_0
### (f * alpha * delta)/(a*b**2+c+f) 
### (f * alpha * delta)/(c+f)

initial_condition = inip.uniform_perturbated

epsilon = 0.5
L = 50
h1 = 0
h2 = 50
arg = [
       [(Pplus1-Pless1)/(L-h1), Pless1 - (Pplus1-Pless1)/(L-h1)*(h1)  , epsilon],
       [ (alpha1/beta1)*(Pplus1-Pless1)/(L-h1),(alpha1/beta1)*(Pless1 - (Pplus1-Pless1)/(L-h1)*h1),epsilon],
       [(Pless2-Pplus2)/h2, Pplus2, epsilon],
       [(alpha2/beta2)*(Pless2-Pplus2)/h2, (alpha2/beta2)*Pplus2, epsilon]
       ]

arg = [
       [0.95, epsilon],
       [(alpha1/beta1)*0.95,epsilon, ],
       [1.975, epsilon],
       [(alpha2/beta2)*(1.975), epsilon]
       ]



if SIMU == 1:
    DF, T = model.sim_PDE(params)


DF, T = read.solution(params["U"])



import plot as plthp

# In[] Solution visualisation
plot = 0
if plot == 1:
    Time_to_plot = list(range(0, len(T), 5))
    
    read.clean()
    
    for t in Time_to_plot:
        plthp.plot2D(DF[t], T[t], "output/heatmaps/plot_at"+str(T[t]), [0, u1_0])

    lim = np.zeros((len(DF)))
    L_x = np.zeros((len(DF), 4))
    for i in range(len(DF)):
        df = DF[i]    
        lim[i]= np.mean(df["u1"]+df["u2"]) - Pplus
        L_x[i] = als.norm_LX(df)
        
    L_x1 = np.sum(L_x, axis = 1)


from matplotlib import gridspec
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
Time_to_plot = [T[0], T[int(len(T)/4)], T[int(len(T)/2)], T[int(3*len(T)/4)], T[-1]]

log_norm = LogNorm(vmin=1e-2, vmax=Pplus)

gs = gridspec.GridSpec(2, len(Time_to_plot)+1, width_ratios= [1]*len(Time_to_plot) + [0.05], wspace=0.3)
ax = {}

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Définir les couleurs pour la palette: rouge, jaune, cyan



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

fig = plt.figure(figsize=(30, 10))
plt.rcParams['text.usetex'] = True
for i in range(len(Time_to_plot)):
    
    df = DF[np.where(T == Time_to_plot[i])[0][0]]
    grid_x, grid_y = np.meshgrid(
        np.linspace(df['x'].min(), df['x'].max(), 200),
        np.linspace(df['y'].min(), df['y'].max(), 200)
    )

    grid_u1 = griddata((df['x'], df['y']), df['u1'], (grid_x, grid_y), method='cubic')
    grid_u2 = griddata((df['x'], df['y']), df['u2'], (grid_x, grid_y), method='cubic')
    
    min_u1 = np.min(grid_u1)
    min_u2 = np.min(grid_u2)
    max_u1 = np.max(grid_u1)
    max_u2 = np.max(grid_u2)
    ax[f'ax_u1{i}'] = plt.subplot(gs[0,i])
    ax[f'ax_u1{i}'].imshow(np.clip(grid_u1,1e-16, Pplus), origin='lower',
                   extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                   cmap='RdYlGn', aspect='auto',
                   vmin=0,
                   vmax=Pplus)
    ax[f'ax_u1{i}'].set_title(r'$u_1(x,y)$', fontweight='bold', fontsize=30)
    ax[f'ax_u1{i}'].set_xticks([])
    ax[f'ax_u1{i}'].set_yticks([])
    
    
    
    ax[f'ax_u2{i}'] = plt.subplot(gs[1,i])
    ax[f'ax_u2{i}'].imshow(np.clip(grid_u2, 1e-16, Pplus), origin='lower',
                   extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                   cmap=custom_cmap , aspect='auto',
                   vmin=0,
                   vmax=Pplus)
    ax[f'ax_u2{i}'].set_title(r'$u_2(x,y)$', fontweight='bold', fontsize=30)
    ax[f'ax_u2{i}'].set_xticks([])
    ax[f'ax_u2{i}'].set_yticks([])
 
for n in range(len(Time_to_plot)):
    for row in [0]:  # 1ère et 3ème ligne
        ax_col = plt.subplot(gs[row, n])
        pos = ax_col.get_position()
        fig.text(
            x=pos.x0 + pos.width / 2,
            y=pos.y1 + 0.05,
            s=r'$t = $'+f" ${Time_to_plot[n]}$",
            ha='center', va='bottom',
            fontsize=25, fontweight='bold'
        )



# Utilise les axes existants, par exemple ceux de la première ligne (u1)
pos_first_row = ax[f'ax_u1{0}'].get_position()
pos_last_col = ax[f'ax_u2{i}'].get_position()
x_center = (pos_first_row.x0 + pos_last_col.x0 + pos_last_col.width) / 2
"""
fig.text(
    x=x_center,
    y=pos_first_row.y1 + 0.1,
    s="Simulation of the scenario with invasion of the boreal forest",
    ha='center', va='bottom',
    fontsize=30, fontweight='bold'
)
"""
        
ax_dummy = fig.add_subplot(111, frameon=False)
ax_dummy.set_visible(False)

A = np.linspace(1e-16, Pplus, 100).reshape(100, 1)
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
cbar_ax0 = plt.subplot(gs[0, -1])
cbar0 = fig.colorbar(im_dummy, cax=cbar_ax0, ticks=ticks)
cbar0.ax.tick_params(labelsize=18)  # Taille des nombres

cbar_ax1 = plt.subplot(gs[1, -1])
cbar1 = fig.colorbar(im_dummy2, cax=cbar_ax1, ticks=ticks)
cbar1.ax.tick_params(labelsize=18)
fig.savefig('output/scenario_invasion_homogeneous.pdf', format='pdf', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.0)
