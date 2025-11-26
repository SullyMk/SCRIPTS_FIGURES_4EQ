# -*- coding: utf-8 -*-
"""

Created on 24/11/2024

@author: Sully Mak, Guillaume Cantin, Benoît Delahaye
Nantes Université - LS2N 

This script contains function to run FreeFem++ scripts for mesh generation and PDE simulation.

"""

import subprocess
import read
import initialpy as inip


# In[14]: functions to run FreeFem++ files

def gen_CI(initial_condition, arg, script_name_mesh_gen = "mesh_gen.edp"):
    """
    Parameters
    ----------
    initial_condition : function
        Function : R^2 -> R^+.
    arg : list of lists
        Parameters of function initial_condition for u0, v0, w0.
    script_name_mesh_gen : string, optional
        File of freefem++ scrip that creates mesh. The default is "mesh_gen.edp".

    Returns
    -------
    U0 : list of mesh + initial conditions.
    """
    
    try:
        subprocess.run(['FreeFem++', script_name_mesh_gen])
    except subprocess.CalledProcessError as e:
        print(f"Erreur {script_name_mesh_gen}: {e}")
        exit()
    
    U0 = inip.read_mesh(initial_condition, arg, mesh_file = "input/mesh.msh")
    
    return U0
        
def load_CI(df, arg, script_name_mesh_gen = "mesh_gen.edp"):
    """
    Parameters
    ----------
    initial_condition : function
        Function : R^2 -> R^+.
    arg : list of lists
        Parameters of function initial_condition for u0, v0, w0.
    script_name_mesh_gen : string, optional
        File of freefem++ scrip that creates mesh. The default is "mesh_gen.edp".

    Returns
    -------
    U0 : list of mesh + initial conditions.
    """
    
    try:
        subprocess.run(['FreeFem++', script_name_mesh_gen])
    except subprocess.CalledProcessError as e:
        print(f"Erreur {script_name_mesh_gen}: {e}")
        exit()
    
    U0 = inip.load_into_mesh(df, mesh_file = "input/mesh.msh")
    
    return U0

def sim_PDE(params, script_name = "modele_mixte.edp"):
    """
    Parameters
    ----------
    params : dict
        Contains parameters of the model.
    script_name : str, optional
        FreeFem++ file for simulation. The default is "modele_gen.edp".

    Returns
    -------
    None.

    """
    
    
    with open("input/params.txt", "w") as file:
        for key, value in params.items():
            file.write(f"{key} {value}\n")
            
    try:
        
        subprocess.run(['FreeFem++', script_name])
        print(f"Script Freefem {script_name} ok")
    except subprocess.CalledProcessError as e:
        print(f"Erreur {script_name}: {e}")
        exit()

    print("Simulation ended")
    filename = params['U']
    DF, T = read.solution(filename)
    return DF, T

