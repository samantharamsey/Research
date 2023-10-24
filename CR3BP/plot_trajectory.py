# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 18:03:51 2023
@author: samantha ramsey
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd

import pickle
# load in constants
pickle_off = open("general_params.pickle", 'rb')
general_params = pickle.load(pickle_off)
# Earth-Moon specific values
pickle_off = open("system_data.pickle", 'rb')
system_data = pickle.load(pickle_off)

import sys
# append CR3BP folder to path so we can call functions from it
path2CR3BP = general_params['path2CR3BP']
sys.path.append(path2CR3BP)

from CR3BP_equations import EOM, EOM_STM, libsolver, jacobi_constant
from tools import normalize, set_axes_equal, get_manifold_ICs
from targeters import xzplane_xfixed
import multiproc_integration as multi


def plot_manifolds_2D(ax, integrated_states, color,
                      return_final_states=True, stable=True):
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    final_states = []
    for i in integrated_states:
        ax.plot(i[0], i[1], c=color, linewidth=0.5, alpha=0.5)
        
        if return_final_states == True:
            statef = [i[j][-1] for j in range(len(i))]
            final_states.append(statef)
    
    if return_final_states == True:
        return final_states
    
def plot_orbit_2D(ax, orbitIC, P, color):
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    ax.plot(sol.y[0],  sol.y[1], c=color)
    ax.plot(sol.y[0], -sol.y[1], c=color)
    
def plot_orbit_velspace_2D(ax, orbitIC, P, color):
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    ax.plot(sol.y[3],  sol.y[4], c=color)

def poincare_section(ax, states, p1, p2, color):
    
    labels = ['x', 'y', 'z', 'v$_x$', 'v$_y$', 'v$_z$']
    ax.grid(alpha=0.2)
    ax.set_xlabel('{} (nd)'.format(labels[p1]))
    ax.set_ylabel('{} (nd)'.format(labels[p2]))
    
    for i in states:
        ax.scatter(i[p1], i[p2], c=color, s=0.5)
    
        
#%%
if __name__ == '__main__':
    
    # define constants
    tol = general_params['tol']
    
    # start and end times for propagation incase no event detected
    t0 = general_params['t0']
    tf = general_params['tf']
    
    # system mass parameter
    mu = system_data['mu']
    # step off distance
    stepoff = system_data['stepoff']
    
    xLib, yLib = libsolver(mu, tol)
    L1 = np.array([xLib[0], yLib[0], 0.0, 0.0, 0.0, 0.0])
    L2 = np.array([xLib[1], yLib[1], 0.0, 0.0, 0.0, 0.0])
    L3 = np.array([xLib[2], yLib[2], 0.0, 0.0, 0.0, 0.0])
    L4 = np.array([xLib[3], yLib[3], 0.0, 0.0, 0.0, 0.0])
    L5 = np.array([xLib[4], yLib[4], 0.0, 0.0, 0.0, 0.0])
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(6)
    phi0 = phi0.reshape(1, 36)
    
    xLib, yLib = libsolver(mu, tol)
    
    trajIC = np.array([ 0.891881109154078, -0.143140830891234, -4.09761503908383e-28, 
                       -0.071026521320225, -0.101158174041878, -1.16657428370783e-27])


    trajIC = np.array([0.8684, 0, 0, 0, -0.221007729, 0])
    trajIC_L1, t_cross = xzplane_xfixed(trajIC, pos2neg=False)
    P_L1 = t_cross
    JC_L1 = jacobi_constant(mu, trajIC_L1)
    
    trajIC = np.array([ 1.1809, 0, 0, 0, -0.1559, 0])
    trajIC_L2, t_cross = xzplane_xfixed(trajIC, pos2neg=False)
    P_L2 = t_cross
    JC_L2 = jacobi_constant(mu, trajIC_L2)
    
    
    # 2D plot formatting
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(alpha=0.2)
    ax1.set_xlabel('x (nd)')
    ax1.set_ylabel('y (nd)')
    
    # integrate from tau_alpha to tau_M
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P_L1), trajIC_L1, 
                    dense_output=True, rtol=tol, atol=tol)
    ax1.plot(sol.y[0], sol.y[1], c='black')
    ax1.plot(sol.y[0], -sol.y[1], c='black')
    
    # integrate from tau_alpha to tau_M
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P_L2), trajIC_L2, 
                    dense_output=True, rtol=tol, atol=tol)
    ax1.plot(sol.y[0],  sol.y[1], c='black')
    ax1.plot(sol.y[0], -sol.y[1], c='black')
    
    ax1.scatter(L1[0], L1[1], marker='*')
    ax1.scatter(L2[0], L2[1], marker='*')
    
    integrated = np.array([sol.y[i][-1] for i in range(len(sol.y))])
    state = integrated[0:6]
    # STM = integrated[6::].reshape(6, 6)
    
    # print('final state: {} \n \n STM: {}'.format(state, STM))
    