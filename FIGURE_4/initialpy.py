# -*- coding: utf-8 -*-
"""

Created on 24/11/2024

@author: Sully Mak, Guillaume Cantin, Benoît Delahaye
Nantes Université - LS2N 

This script contains function to initialize the mesh and the initial conditions.

"""
import sys
import numpy as np
import subprocess
import pandas as pd
import random
from scipy.interpolate import griddata
# In[14] : functions to manage mesh file

def read_mesh(initial_condition, arg, mesh_file = "input/mesh.msh"):
    """
    Parameters
    ----------
    initial_condition : function
        Function : R^2 -> R^2.
    arg : list of lists
        Parameters of function initial_condition for u0, v0, w0.
    mesh_file : string, optional
        File name of the mesh generated with mesh_gen.edp. The default is "input/mesh.msh".

    Returns
    -------
    nodes : list of mesh + initial conditions.

    """
    try:
        subprocess.run(['FreeFem++', "mesh_gen.edp"])
    except subprocess.CalledProcessError as e:
        exit()
    with open(mesh_file, "r") as f:
        lines = f.readlines()

    # Extraire les informations du maillage
    header = list(map(int, lines[0].split()))
    n_nodes = header[0]
    
    # Extraire les coordonnées des nœuds
    nodes = []
    for i in range(1, n_nodes + 1):
        parts = list(map(float, lines[i].split()))
        x, y = parts[0], parts[1]
        nodes.append((x, y, initial_condition(arg[0], x, y), initial_condition(arg[1], x, y), initial_condition(arg[2], x, y), initial_condition(arg[3], x, y)))
    np.savetxt("input/U0.txt", nodes, fmt="%.6f", comments="")
    
    U0 = np.array(nodes)
    colnames = ["x", "y", "um(x,y)", "wm(x,y)", "ub(x,y)", "wb(x,y)"]
    U0 = pd.DataFrame(U0, columns  = colnames)
    
    return U0



def load_into_mesh(df, filename = "input/U0.txt", mesh_file = "input/mesh.msh"):
    """
    Parameters  
    ----------
    initial_condition : function
        Function : R^2 -> R^2.
    arg : list of lists
        Parameters of function initial_condition for u0, v0, w0.
    mesh_file : string, optional
        File name of the mesh generated with mesh_gen.edp. The default is "input/mesh.msh".

    Returns
    -------
    nodes : list of mesh + initial conditions.

    """
    
    
    grid_x, grid_y = np.mgrid[min(df["x"]):max(df["x"]):100j, min(df["y"]):max(df["y"]):100j]
    
    points = np.column_stack((df["x"], df["y"]))
    grid_u1 = griddata(points, df["u1"], (grid_x, grid_y), method='cubic')
    grid_w1 = griddata(points, df["w1"], (grid_x, grid_y), method='cubic')
    grid_u2 = griddata(points, df["u2"], (grid_x, grid_y), method='cubic')
    grid_w2 = griddata(points, df["w2"], (grid_x, grid_y), method='cubic')
    
    def interpolate(f, x_val, y_val):
        return griddata(np.column_stack((df["x"], df["y"])), f, (x_val, y_val), method='cubic')

    
    
    try:
        subprocess.run(['FreeFem++', "mesh_gen.edp"])
    except subprocess.CalledProcessError as e:
        exit()
    with open(mesh_file, "r") as f:
        lines = f.readlines()


    # Extraire les informations du maillage
    header = list(map(int, lines[0].split()))
    n_nodes = header[0]
    
    # Extraire les coordonnées des nœuds
    nodes = []
    for i in range(1, n_nodes + 1):
        parts = list(map(float, lines[i].split()))
        x, y = parts[0], parts[1]
        u1 = float(interpolate(df["u1"], x, y))
        w1 = float(interpolate(df["w1"], x, y))
        u2 = float(interpolate(df["u2"], x, y))
        w2 = float(interpolate(df["w2"], x, y))
        nodes.append((x, y, u1, w1, u2, w2))
    np.savetxt(filename, nodes, fmt="%.6f", comments="")
    
    U0 = np.array(nodes)
    colnames = ["x", "y", "um(x,y)", "wm(x,y)", "ub(x,y)", "wb(x,y)"]
    U0 = pd.DataFrame(U0, columns  = colnames)
    
    return U0

# In[54]: function in order to set initials conditions

def uniform(arg,x,y):
    """
    Parameters
    ----------
    arg : Float
        f(x,y) = arg.
    x : float
        x axis position.
    y : float
        y axis position.

    Returns
    -------
    arg : Float
        f(x,y) = arg.

    """
    return arg

def uniform_perturbated(arg, x, y):
    
    return arg[0] + (2*np.random.rand()-1)*arg[1] 

def disk(arg, x, y):
    """
    Parameters
    ----------
    arg : LIST
        radius, xc, yc, z
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if ((x - arg[1])*(x - arg[1] )+(y- arg[2])*(y- arg[2])< arg[0]*arg[0]):
        return arg[3]
    else:
        return 0
    
def fullrandom(arg, x, y):
    return np.random.rand()*arg 



def regions(arg, x, y):
    if(x > arg[0])and(x < arg[1])and(y>arg[2])and(y<arg[3]):
        return arg[4]
    else:
        return 0
    
def regions_noise(arg, x, y):
    if(x >= arg[0])and(x <= arg[1])and(y>=arg[2])and(y<=arg[3]):
        return arg[4] + (2*np.random.rand()-1)*arg[5] 
    else:
        return 0
    
def half_and_fractured(arg, x, y):
    if(x >= arg[0])and(x <= arg[1])and(y>=arg[2])and(y<=arg[3]):
        return arg[4] + (2*np.random.rand()-1)*arg[5] 
    else:
        return arg[6] + (2*np.random.rand()-1)*arg[7]
    
def pente(arg, x ,y):
    
    return np.max([0,np.min([arg[3], (y * arg[0] + arg[1]) - ((np.random.rand())*arg[2]*(y * arg[0] + arg[1]))])]) 
    


def piecewise_function(arg, x, y):
    """
    Calcule la valeur de la fonction définie par morceaux :
    - f(x) = A pour x ≤ h
    - f(x) = ax + b pour h < x ≤ L avec a et b tels que :
        a*h + b = A
        a*L + b = B
    - f(x) = B pour x > L
    """
    # Calcul des coefficients de l'interpolation affine entre (h, A) et (L, B)
    A = arg[0] 
    B = arg[1] 
    L1 = arg[2] 
    L2 = arg[3] 
    
    a = (A - B) / (L1 - L2)
    b = A - a * L1

    # Définition de f(x) selon les intervalles
    if y <= L1:
        return np.max([0, A+  ((2*np.random.rand()-1) * arg[4])*A])
    elif y <= L2:
        return np.max([0, (y * a + b) +  ((2*np.random.rand()-1) * arg[4])*(y * a + b) ])
    else:
        return np.max([0, B +  ((2*np.random.rand()-1) * arg[4])*B])
    