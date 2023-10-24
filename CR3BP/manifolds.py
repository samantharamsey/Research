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

from CR3BP_equations import EOM, EOM_STM, libsolver
from tools import normalize, set_axes_equal, get_manifold_ICs
from targeters import xzplane_xfixed, xyplane_vzfixed
import multiproc_integration as multi


def plot_manifolds_2D(ax, integrated_states, 
                      return_final_states=True, stable=True):
    
    if stable==True:
        color = 'blue'
    else:
        color = 'red'
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    final_states = []
    for i in integrated_states:
        ax.plot(i[0], i[1], c=color, linewidth=0.5)
        
        if return_final_states == True:
            statef = [i[j][-1] for j in range(len(i))]
            final_states.append(statef)
    
    if return_final_states == True:
        return final_states
    
def plot_manifolds_3D(ax, integrated_states, 
                      return_final_states=True, stable=True):
    pad = 10
    if stable==True:
        color = 'blue'
    else:
        color = 'red'
        
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)', labelpad=pad)
    ax.set_ylabel('y (nd)', labelpad=pad)
    ax.set_zlabel('z (nd)', labelpad=pad)
    
    final_states = []
    for i in integrated_states:
        ax.plot(i[0], i[1], i[2], c=color, linewidth=0.3)
        
        if return_final_states == True:
            statef = [i[j][-1] for j in range(len(i))]
            final_states.append(statef)
    
    if return_final_states == True:
        return final_states
    
        

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
    
    # libration point locations
    xLib, yLib = libsolver(mu, tol)
    
    # Lyapunov and Halo orbits (xz plane targeting)
    # LyapIC = np.array([0.9569, 0.0, 0.0, 0.0, -0.8839, 0.0])
    # LyapIC = np.array([0.8879, 0.0, 0.0, 0.0, -0.3357, 0.0])
    # LyapIC = np.array([0.8554, 0.0, 0.0, 0.0, -0.1379, 0.0])
    # HaloIC = np.array([0.82377, 0.0, 0.04850, 0.0, 0.15808, 0.0])
    
    # Axial orbit (xyplane targeting)
    AxialIC = np.array([0.79962863, 0.0, 0.0, 0.0, 0.37169140, 0.23241799])
    AxialIC = np.array([0.81217958, 0.0, 0.0, 0.0, 0.32101575, 0.29643532])
    AxialIC = np.array([0.83709449, 0.0, 0.0, 0.0, 0.21479953, 0.38371514])

    corr_state, t_cross = xyplane_vzfixed(AxialIC)
    
    num_manifolds = 50
    stableIC, tpoints = get_manifold_ICs(mu, corr_state, t_cross*2, stepoff, num_manifolds, 
                               positive_dir=True, stable=True)
    unstableIC, tpoints = get_manifold_ICs(mu, corr_state, t_cross*2, stepoff, num_manifolds, 
                               positive_dir=True, stable=False, full_state=True)
    
    # # plot style
    # # plt.style.use('dark_background')
    # plt.rcParams['grid.linewidth'] = 0.2
    # plt.locator_params(axis='both', nbins=4)
    
    # 2D plot formatting
    # fig1, ax1 = plt.subplots()
    
    # 3D plot formatting
    fig1 = plt.figure()
    ax1  = fig1.add_subplot(111, projection= '3d')
    
    # determine total number of processing cores on machine
    ncores = int(mp.cpu_count()/2)
    
    # multiprocess to propagate manifolds
    p = mp.Pool(ncores)
    
    #%%
    # integrate stable manifolds to Moon and get final state at Lunar crossing
    integ_states_to_moon = p.map(multi.stable_to_tf, stableIC)
    # index state portion of solve_ivp solution
    stable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # plot and get final state at Lunar crossing
    stable_states_at_Moon = plot_manifolds_3D(ax1, stable_states_to_moon)
    
    # # integrate stable manifolds from Lunar crossing to xaxis crossing
    # integ_states_to_xaxis = p.map(multi.stable_to_xaxis, stable_states_at_Moon)
    # # index state portion of solve_ivp solution
    # stable_states_to_xaxis = [integ_states_to_xaxis[i].y for i in range(len(integ_states_to_xaxis))]
    # # plot and get final state at x-axis crossing
    # stable_states_at_xaxis = plot_manifolds_3D(ax1, stable_states_to_xaxis)
    
    #%%
    # integrate unstable manifolds to Moon and get final state at Lunar crossing
    # integ_states_to_moon = p.map(multi.unstable_to_tf, unstableIC)
    # # index state portion of solve_ivp solution
    # unstable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # # plot and get final state at Lunar crossing
    # unstable_states_at_Moon = plot_manifolds_3D(ax1, unstable_states_to_moon, stable=False)
    
    # # integrate stable manifolds from Lunar crossing to xaxis crossing
    # integ_states_to_xaxis = p.map(multi.unstable_to_xaxis, unstable_states_at_Moon)
    # # index state portion of solve_ivp solution
    # unstable_states_to_xaxis = [integ_states_to_xaxis[i].y for i in range(len(integ_states_to_xaxis))]
    # # plot and get final state at x-axis crossing
    # unstable_states_at_xaxis = plot_manifolds_3D(ax1, unstable_states_to_xaxis, stable=False)
    
    #%%
    
    ax1.set_title('Stable and Unstable Manifolds in Earth-Moon System')
    # plot the moon
    rmoon = system_data['r moon']
    
    # 2D moon
    # moon = plt.Circle((1-mu, 0), rmoon, color='gray', zorder=3, alpha=0.8)
    # ax1.add_patch(moon)
    
    # 3D moon
    r = rmoon
    # sphere math
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0,   np.pi, 100)
    x = r*np.outer(np.cos(u), np.sin(v)) + (1-mu)
    y = r*np.outer(np.sin(u), np.sin(v))
    z = r*np.outer(np.ones(np.size(u)), np.cos(v))
    # plot the surface
    ax1.plot_surface(x, y, z, color='gray', alpha=0.5)
    
    # 3D axes formatting has to come after the plot
    set_axes_equal(ax1)
    ax1.grid(alpha=0.1) 
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.set_box_aspect(aspect=None, zoom=1.2)
    
    
    # for i in range(num_manifolds):
    #     ax1.scatter(unstable_states_to_moon[i][0][0],
    #                 unstable_states_to_moon[i][1][0],
    #                 unstable_states_to_moon[i][2][0])
    
    
    
    
    
    
    
    
    















