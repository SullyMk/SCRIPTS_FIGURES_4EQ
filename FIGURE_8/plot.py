# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 13:56:12 2025

@author: e24h297n
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os, shutil

def plot2D(df, t, name, extremum = [0, None]):
    """
    Parameters
    ----------
    df : list of Dataframe that contains solution at indice corresponding to times
    times : list of values of the discretised time
    t : int
        index of time simulation.
    name : str
        name for the figure generated.

    Returns
    -------
    None.

    """
    
    vmin, vmax = extremum[0], extremum[1]
    # Définir une grille régulière pour l'interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(df['x'].min(), df['x'].max(), 200),
        np.linspace(df['y'].min(), df['y'].max(), 200)
    )
    
    # Interpolation avec scipy.griddata
    grid_u1 = griddata(
        (df['x'], df['y']),    # Coordonnées des nœuds
        df['u1'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    grid_u2 = griddata(
        (df['x'], df['y']),    # Coordonnées des nœuds
        df['u2'],          # Valeurs à interpoler
        (grid_x, grid_y),      # Points de la grille régulière
        method='cubic'         # Interpolation cubique (smooth)
    )
    # Affichage de la carte de chaleur
    plt.figure(figsize=(8, 6))
    heatmap = plt.contourf(grid_x, grid_y, grid_u1, levels=100,  cmap='RdYlGn', vmin=vmin, vmax=vmax)
    plt.colorbar(heatmap, label='u(x, y)')
    
    # Ajouter les points originaux pour comparaison
    #plt.scatter(df['x'], df['y'], c='red', label='Points originaux', s=30)
    plt.title('u1('+str(t)+', x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(name+"mixte.png",  dpi=300)

    plt.figure(figsize=(8, 6))
    heatmap = plt.contourf(grid_x, grid_y, grid_u2, levels=100,  cmap='RdYlGn', vmin=vmin, vmax=vmax)
    plt.colorbar(heatmap, label='u(x, y)')
    
    # Ajouter les points originaux pour comparaison
    #plt.scatter(df['x'], df['y'], c='red', label='Points originaux', s=30)
    plt.title('u2('+str(t)+', x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(name+"boreal.png",  dpi=300)
    
    
    
    
    
def plot_solution(df, t, name):
    grid_x, grid_y = np.meshgrid(
    np.linspace(df['x'].min(), df['x'].max(), 200),
    np.linspace(df['y'].min(), df['y'].max(), 200)
    )
    grid_u1 = griddata(df[['x', 'y']].values, df['u1(x,y)'].values, (grid_x, grid_y), method='linear')

