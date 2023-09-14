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
from tools import normalize, set_axes_equal
from targeters import xzplane_xfixed
import multiproc_integration as multi


def get_manifold_IC(mu, orbitIC, P, stepoff, num_manifolds, 
                    positive_dir=True, stable=True):
    
    # need set of discreetized points around orbit
    tpoints = np.linspace(t0, P, num_manifolds, endpoint=False)
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(6)
    phi0 = phi0.reshape(1, 36)
    
    # add STM to state
    orbitIC = np.concatenate([orbitIC, phi0[0]])
    
    # integrate initial conditions for one full period
    sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # extract final values from integration results
    phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
    
    # monodromy matrix
    monod  = phi[6::].reshape((6, 6))
    
    # get eigenvalues nad vectors of monodromy matrix
    vals, vecs = np.linalg.eig(monod)
    
    # create a list of the eigenvalue indicies in ascending order
    idxs = list(range(0, 6))
    idxs.sort(key = lambda x:np.abs(vals[x]))
    
    # stable eigenvalue with be the one with the smallest magnitude
    stble_vec = np.real(vecs[:, idxs[ 0]])
    unstb_vec = np.real(vecs[:, idxs[-1]])
    
    # integrate initial conditions for one full period
    sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, t_eval=tpoints, rtol=tol, atol=tol)
    
    # make a list of states and STMs at each point
    states, STMs = [], []
    for j in range(num_manifolds):
        # extract final values from integration results
        phi = np.array([sol.y[i][j] for i in range(len(sol.y))])
        states.append(phi[0:6])
        STMs.append(phi[6::].reshape((6, 6)))
    
    # list of initial conditions for step off onto manifold
    manifoldIC = []
    
    for i, STM in enumerate(STMs):
        
        # make a copy of the state at the current point in the orbit
        state = np.copy(states[i])
        
        # floquet theory to transition the eigenvectors
        st_vec = STM@stble_vec
        un_vec = STM@unstb_vec
        
        # perturbation from orbit onto stable/unstable eigenvector
        if stable:
            pert = stepoff*(st_vec/np.linalg.norm(st_vec[0:3]))
        else:
            pert = stepoff*(un_vec/np.linalg.norm(un_vec[0:3]))
        
        # positive direction
        if positive_dir:
            if pert[0] > 0:
                state[0:6] = state[0:6] + pert
            else:
                state[0:6] = state[0:6] - pert
        # negative direction
        else:
            if pert[0] < 0:
                state[0:6] = state[0:6] + pert
            else:
                state[0:6] = state[0:6] - pert
        
        manifoldIC.append(state)
    return manifoldIC, tpoints

def get_manifold_IC2(mu, orbitIC, P, stepoff, num_manifolds, 
                    positive_dir=True, stable=True):
    
    # need set of discreetized points around orbit
    tpoints = np.linspace(t0, P, num_manifolds, endpoint=False)
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(6)
    phi0 = phi0.reshape(1, 36)
    
    # add STM to state
    orbitIC = np.concatenate([orbitIC, phi0[0]])
    
    # integrate initial conditions for one full period
    sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # extract final values from integration results
    phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
    
    # monodromy matrix
    monod  = phi[6::].reshape((6, 6))
    
    # get eigenvalues nad vectors of monodromy matrix
    vals, vecs = np.linalg.eig(monod)
    
    # create a list of the eigenvalue indicies in ascending order
    idxs = list(range(0, 6))
    idxs.sort(key = lambda x:np.abs(vals[x]))
    
    # stable eigenvalue with be the one with the smallest magnitude
    stble_vec = np.real(vecs[:, idxs[ 0]])
    unstb_vec = np.real(vecs[:, idxs[-1]])
    
    # integrate initial conditions for one full period
    sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, t_eval=tpoints, rtol=tol, atol=tol)
    
    # make a list of states and STMs at each point
    states, STMs = [], []
    for j in range(num_manifolds):
        # extract final values from integration results
        phi = np.array([sol.y[i][j] for i in range(len(sol.y))])
        states.append(phi[0:6])
        STMs.append(phi[6::].reshape((6, 6)))
    
    # list of initial conditions for step off onto manifold
    manifoldIC = []
    
    for i, STM in enumerate(STMs):
        
        # make a copy of the state at the current point in the orbit
        state = np.copy(states[i])
        
        # floquet theory to transition the eigenvectors
        st_vec = STM@stble_vec
        un_vec = STM@unstb_vec
        
        # perturbation from orbit onto stable/unstable eigenvector
        if stable:
            pert = stepoff*(st_vec/np.linalg.norm(st_vec))
        else:
            pert = stepoff*(un_vec/np.linalg.norm(un_vec))
        
        # positive direction
        if positive_dir:
            if pert[0] > 0:
                state[0:6] = state[0:6] + pert
            else:
                state[0:6] = state[0:6] - pert
        # negative direction
        else:
            if pert[0] < 0:
                state[0:6] = state[0:6] + pert
            else:
                state[0:6] = state[0:6] - pert
        
        manifoldIC.append(state)
    return manifoldIC, tpoints


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
    
    xLib, yLib = libsolver(mu, tol)
    L1 = np.array([xLib[0], yLib[0], 0.0, 0.0, 0.0, 0.0])
    L2 = np.array([xLib[1], yLib[1], 0.0, 0.0, 0.0, 0.0])
    L3 = np.array([xLib[2], yLib[2], 0.0, 0.0, 0.0, 0.0])
    L4 = np.array([xLib[3], yLib[3], 0.0, 0.0, 0.0, 0.0])
    L5 = np.array([xLib[4], yLib[4], 0.0, 0.0, 0.0, 0.0])
    
    LyapIC = np.array([0.9569, 0.0, 0.0, 0.0, -0.8839, 0.0])
    LyapIC = np.array([0.8879, 0.0, 0.0, 0.0, -0.3357, 0.0])
    LyapIC = np.array([0.8554, 0.0, 0.0, 0.0, -0.1379, 0.0])
    
    HaloIC = np.array([0.82377, 0.0, 0.04850, 0.0, 0.15808, 0.0])
    
    corr_state, t_cross = xzplane_xfixed(HaloIC)
    
    num_manifolds = 50
    stableIC, tpoints = get_manifold_IC(mu, corr_state, t_cross*2, stepoff, num_manifolds, 
                               positive_dir=True, stable=True)
    unstableIC, tpoints = get_manifold_IC(mu, corr_state, t_cross*2, stepoff, num_manifolds, 
                               positive_dir=True, stable=False)
    
    # plot style
    # plt.style.use('dark_background')
    plt.rcParams['grid.linewidth'] = 0.2
    plt.locator_params(axis='both', nbins=4)
    
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
    integ_states_to_moon = p.map(multi.stable_to_secondary, stableIC)
    # index state portion of solve_ivp solution
    stable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # plot and get final state at Lunar crossing
    stable_states_at_Moon = plot_manifolds_3D(ax1, stable_states_to_moon)
    
    # integrate stable manifolds from Lunar crossing to xaxis crossing
    integ_states_to_xaxis = p.map(multi.stable_to_xaxis, stable_states_at_Moon)
    # index state portion of solve_ivp solution
    stable_states_to_xaxis = [integ_states_to_xaxis[i].y for i in range(len(integ_states_to_xaxis))]
    # plot and get final state at x-axis crossing
    stable_states_at_xaxis = plot_manifolds_3D(ax1, stable_states_to_xaxis)
    
    #%%
    # integrate unstable manifolds to Moon and get final state at Lunar crossing
    integ_states_to_moon = p.map(multi.unstable_to_secondary, unstableIC)
    # index state portion of solve_ivp solution
    unstable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # plot and get final state at Lunar crossing
    unstable_states_at_Moon = plot_manifolds_3D(ax1, unstable_states_to_moon, stable=False)
    
    # integrate stable manifolds from Lunar crossing to xaxis crossing
    integ_states_to_xaxis = p.map(multi.unstable_to_xaxis, unstable_states_at_Moon)
    # index state portion of solve_ivp solution
    unstable_states_to_xaxis = [integ_states_to_xaxis[i].y for i in range(len(integ_states_to_xaxis))]
    # plot and get final state at x-axis crossing
    unstable_states_at_xaxis = plot_manifolds_3D(ax1, unstable_states_to_xaxis, stable=False)
    
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
    
    
    
    
    
    
    
    
    
    















