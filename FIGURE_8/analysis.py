# -*- coding: utf-8 -*-
"""

Created on 24/11/2024

@author: Sully Mak, Guillaume Cantin, Benoît Delahaye
Nantes Université - LS2N 

This script contains function for computing norms 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import griddata
import pickle as pic
from scipy.integrate import trapezoid


def l2_norm(u, dx, dy):
    return (np.nansum(np.abs(u)**2) * dx * dy)

def expo(t, Lambda, C):
    return C * np.exp(-Lambda*t)

def logi(t, L, k, t0):
    return L/(1+np.exp(-k*(t-t0)))

def J(val, T, Norm):
    J = 0
    for i in range(len(T)):

        J = J + (Norm[i]-logi(T[i], val[0], val[1], val[2]))**2

    return J

def int_trpz(u, x, y):
    integral_x = trapezoid(u, x[0,:])     # Intégration selon x pour chaque ligne y
    I = trapezoid(integral_x, y[:,0])      # Puis intégration du résultat selon y
    return I

import model
import read
    
    
    
    
def norm_LX(U1, U2 = [0, 0, 0, 0]):
    """
    Parameters
    ----------

    U1 : Dataframe pandas
        Solution computed from freefem++ at time t.
    U2 : Dataframe pandas
        Solution computed from freefem++ at time t.

    Returns
    -------
    N : list of arrays
        contains norm between U1 and U1.

    """
        
    
    grid_x, grid_y = np.meshgrid(
        np.linspace(U1['x'].min(), U1['x'].max(), 100),
        np.linspace(U1['y'].min(), U1['y'].max(), 100))
    dx = np.mean(np.diff(grid_x))  # Approximation du pas en x
    dy = np.mean(np.diff(grid_y, axis = 0))  # Approximation du pas en y
    grid_ub = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['u1'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    grid_wb = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['w1'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    grid_up = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['u2'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    grid_wp = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['w2'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    
    UVW1 = [grid_ub, grid_wb, grid_up, grid_wp]
    if isinstance(U2, pd.DataFrame):
        grid_ub2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['u1'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        grid_wb2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['w1'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        grid_up2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['u2'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        grid_wp2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['w2'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        
        UVW2 = [grid_ub2, grid_wb2, grid_up2, grid_wp2]
   
        N = [UVW1[i] - UVW2[i] for i in range(len(UVW1)) ]
        N = [np.sqrt(int_trpz(N[i], grid_x, grid_y)**2) for i in range(len(UVW1)) ]
        
    if isinstance(U2, list) : 
        N = [np.sqrt(int_trpz(UVW1[i]-U2[i], grid_x, grid_y)**2) for i in range(len(UVW1)) ]
    return N




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:37:25 2025

@author: e24h297n
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import griddata
import pickle as pic


def l2_norm(u, dx, dy):
    return (np.nansum(np.abs(u)**2) * dx * dy)

def expo(t, Lambda, C):
    return C * np.exp(-Lambda*t)

def logi(t, L, k, t0):
    return L/(1+np.exp(-k*(t-t0)))

def J(val, T, Norm):
    J = 0
    for i in range(len(T)):

        J = J + (Norm[i]-logi(T[i], val[0], val[1], val[2]))**2

    return J

import model
import read
    
    
    
    
def norm_L2(U1, U2 = [0, 0, 0, 0]):
    """
    Parameters
    ----------

    U1 : Dataframe pandas
        Solution computed from freefem++ at time t.
    U2 : Dataframe pandas
        Solution computed from freefem++ at time t.

    Returns
    -------
    N : list of arrays
        contains norm between U1 and U1.

    """
        
    
    grid_x, grid_y = np.meshgrid(
        np.linspace(U1['x'].min(), U1['x'].max(), 100),
        np.linspace(U1['y'].min(), U1['y'].max(), 100))
    dx = np.mean(np.diff(grid_x))  # Approximation du pas en x
    dy = np.mean(np.diff(grid_y, axis = 0))  # Approximation du pas en y
    

    
    
        
        
        
    grid_ub = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['u1'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    grid_wb = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['w1'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    grid_up = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['u2'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    grid_wp = griddata(
        (U1['x'], U1['y']),    # Coordonnées des nœuds
        U1['w2'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    
    UVW1 = [grid_ub, grid_wb, grid_up, grid_wp]
    if isinstance(U2, pd.DataFrame):
        grid_ub2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['u1'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        grid_wb2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['w1'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        grid_up2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['u2'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        grid_wp2 = griddata(
            (U2['x'], U2['y']),    # Coordonnées des nœuds
            U2['w2'],          # Valeurs à interpoler
            (grid_x, grid_y),      # Points de la grille régulière
            method='cubic'         # Interpolation cubique (smooth)
        )
        
        UVW2 = [grid_ub2, grid_wb2, grid_up2, grid_wp2]
   
        N = [UVW1[i] - UVW2[i] for i in range(len(UVW1)) ]
        N = [int_trpz(N[i], dx, dy) for i in range(len(UVW1)) ]
        
    if isinstance(U2, list) : 
        N = [int_trpz(UVW1[i]-U2[i], dx, dy) for i in range(len(UVW1)) ]
    return N




