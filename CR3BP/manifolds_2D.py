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

from CR3BP_equations import EOM_2D, EOM_STM_2D, libsolver
from tools import normalize, set_axes_equal
from targeters import xzplane_xfixed
import multiproc_integration as multi


def get_manifold_IC_2D(mu, orbitIC, P, stepoff, num_manifolds, 
                    positive_dir=True, stable=True):
    
    # need set of discreetized points around orbit
    tpoints = np.linspace(t0, P, num_manifolds, endpoint=False)
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(4)
    phi0 = phi0.reshape(1, 16)
    
    # add STM to state
    orbitIC = np.concatenate([orbitIC, phi0[0]])
    
    # integrate initial conditions for one full period
    sol = solve_ivp(lambda t, y: EOM_STM_2D(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # extract final values from integration results
    phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
    
    # monodromy matrix
    monod  = phi[4::].reshape((4, 4))
    
    # get eigenvalues nad vectors of monodromy matrix
    vals, vecs = np.linalg.eig(monod)
    
    # create a list of the eigenvalue indicies in ascending order
    idxs = list(range(0, 4))
    idxs.sort(key = lambda x:np.abs(vals[x]))
    
    # stable eigenvalue with be the one with the smallest magnitude
    stble_vec = np.real(vecs[:, idxs[ 0]])
    unstb_vec = np.real(vecs[:, idxs[-1]])
    
    # integrate initial conditions for one full period
    sol = solve_ivp(lambda t, y: EOM_STM_2D(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, t_eval=tpoints, rtol=tol, atol=tol)
    
    # make a list of states and STMs at each point
    states, STMs = [], []
    for j in range(num_manifolds):
        # extract final values from integration results
        phi = np.array([sol.y[i][j] for i in range(len(sol.y))])
        states.append(phi[0:4])
        STMs.append(phi[4::].reshape((4, 4)))
    
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
            pert = stepoff*(st_vec/np.linalg.norm(st_vec[0:2]))
        else:
            pert = stepoff*(un_vec/np.linalg.norm(un_vec[0:2]))
        
        # positive direction
        if positive_dir:
            if pert[0] > 0:
                state[0:4] = state[0:4] + pert
            else:
                state[0:4] = state[0:4] - pert
        # negative direction
        else:
            if pert[0] < 0:
                state[0:4] = state[0:4] + pert
            else:
                state[0:4] = state[0:4] - pert
        
        manifoldIC.append(state)
    return manifoldIC

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
    
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    ax.plot(sol.y[0],  sol.y[1], c=color)
    ax.plot(sol.y[0], -sol.y[1], c=color)
    
def plot_orbit_velspace_2D(ax, orbitIC, P, color):
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, P), orbitIC, 
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
    L1 = np.array([xLib[0], yLib[0], 0.0, 0.0])
    L2 = np.array([xLib[1], yLib[1], 0.0, 0.0])
    L3 = np.array([xLib[2], yLib[2], 0.0, 0.0])
    L4 = np.array([xLib[3], yLib[3], 0.0, 0.0])
    L5 = np.array([xLib[4], yLib[4], 0.0, 0.0])
    
    LyapIC = np.array([0.9569, 0.0, 0.0, -0.8839])
    LyapIC = np.array([0.8879, 0.0, 0.0, -0.3357])
    LyapIC = np.array([0.8554, 0.0, 0.0, -0.1379])
    
    corr_state, t_cross = xzplane_xfixed(LyapIC)
    
    num_manifolds = 200
    stableIC1 = get_manifold_IC_2D(mu, corr_state, t_cross*2, stepoff, num_manifolds, 
                               positive_dir=True, stable=True)
    unstableIC = get_manifold_IC_2D(mu, corr_state, t_cross*2, stepoff, num_manifolds, 
                                positive_dir=True, stable=False)
    
    # plot style
    # plt.style.use('dark_background')
    plt.rcParams['grid.linewidth'] = 0.2
    plt.locator_params(axis='both', nbins=4)
    
    # 2D plot formatting
    fig1, ax1 = plt.subplots()
    
    # determine total number of processing cores on machine
    ncores = int(mp.cpu_count()/2)
    
    # multiprocess to propagate manifolds
    p = mp.Pool(ncores)
    
    #%%
    # integrate stable manifolds to Moon and get final state at Lunar crossing
    integ_states_to_moon = p.map(multi.stable_to_secondary, stableIC1)
    # index state portion of solve_ivp solution
    stable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # plot and get final state at Lunar crossing
    stable_states_at_Moon = plot_manifolds_2D(ax1, stable_states_to_moon, 'blue')
    
    # integrate stable manifolds from Lunar crossing to xaxis crossing
    integ_states_to_xaxis = p.map(multi.stable_to_xaxis, stable_states_at_Moon)
    # index state portion of solve_ivp solution
    stable_states_to_xaxis = [integ_states_to_xaxis[i].y for i in range(len(integ_states_to_xaxis))]
    # plot and get final state at x-axis crossing
    stable_states_at_xaxis = plot_manifolds_2D(ax1, stable_states_to_xaxis, 'blue')
    
    #%%
    # integrate stable manifolds to Moon and get final state at Lunar crossing
    integ_states_to_moon = p.map(multi.unstable_to_secondary, unstableIC)
    # index state portion of solve_ivp solution
    unstable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # plot and get final state at Lunar crossing
    unstable_states_at_Moon = plot_manifolds_2D(ax1, unstable_states_to_moon, 'red')
    
    # integrate stable manifolds from Lunar crossing to xaxis crossing
    integ_states_to_xaxis = p.map(multi.unstable_to_xaxis, unstable_states_at_Moon)
    # index state portion of solve_ivp solution
    unstable_states_to_xaxis = [integ_states_to_xaxis[i].y for i in range(len(integ_states_to_xaxis))]
    # plot and get final state at x-axis crossing
    unstable_states_at_xaxis = plot_manifolds_2D(ax1, unstable_states_to_xaxis, 'red')
    
    #%%
    
    ax1.set_title('Stable and Unstable Manifolds in Earth-Moon System')
    # plot the moon
    rmoon = system_data['r moon']
    
    # 2D moon
    moon = plt.Circle((1-mu, 0), rmoon, color='gray', zorder=3, alpha=0.8)
    ax1.add_patch(moon)
    
    #%%
    fig2, ax2 = plt.subplots()
    poincare_section(ax2,   stable_states_at_xaxis, 0, 3, 'blue')
    poincare_section(ax2, unstable_states_at_xaxis, 0, 3, 'red')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # # 3D moon
    # r = rmoon
    # # sphere math
    # u = np.linspace(0, 2*np.pi, 100)
    # v = np.linspace(0,   np.pi, 100)
    # x = r*np.outer(np.cos(u), np.sin(v)) + (1-mu)
    # y = r*np.outer(np.sin(u), np.sin(v))
    # z = r*np.outer(np.ones(np.size(u)), np.cos(v))
    # # plot the surface
    # ax1.plot_surface(x, y, z, color='gray', alpha=0.5)
    
    # # 3D axes formatting has to come after the plot
    # set_axes_equal(ax1)
    # ax1.grid(alpha=0.1) 
    # ax1.xaxis.pane.fill = False
    # ax1.yaxis.pane.fill = False
    # ax1.zaxis.pane.fill = False
    # ax1.set_box_aspect(aspect=None, zoom=1.2)

    
    # # 2D plot formatting
    # fig2, ax2 = plt.subplots()
    
    
    # x1s, y1s = [], []
    # x2s, y2s = [], []
    # x3s, y3s = [], []
    # for i in range(len(stableIC1)):
    #     x1s.append(stableIC1[i][0])
    #     y1s.append(stableIC1[i][1])
    #     x2s.append(stableIC2[i][0])
    #     y2s.append(stableIC2[i][1])
    #     x3s.append(stableIC3[i][0])
    #     y3s.append(stableIC3[i][1])
    
    # ax2.scatter(x1s, y1s, color='magenta')
    # ax2.scatter(x2s, y2s, color='cyan')
    # ax2.scatter(x3s, y3s, color='goldenrod')
    # plot_orbit_2D(ax2, corr_state, t_cross, 'black')
    # ax2.legend(['normalized by position', 'normalized by entire state', 'normalized by position, stepoff/2',  'Lyapunov orbit'])
    # ax2.set_title('position space')
    
    # for i in range(len(stableIC1)):
    #     vec = normalize(stvecs2[i][0:3])*0.0001
    #     ax2.plot([states2[i][0], states2[i][0]+vec[0]], [states2[i][1], states2[i][1]+vec[1]], c='blue')
        
    # #%%
    
    # # 2D plot formatting
    # fig3, ax3 = plt.subplots()
    
    # x1s, y1s = [], []
    # x2s, y2s = [], []
    # for i in range(len(stableIC1)):
    #     x1s.append(stableIC1[i][3])
    #     y1s.append(stableIC1[i][4])
    #     x2s.append(stableIC2[i][3])
    #     y2s.append(stableIC2[i][4])
    
    # ax3.scatter(x1s, y1s, color='magenta')
    # ax3.scatter(x2s, y2s, color='cyan')
    # plot_orbit_velspace_2D(ax3, corr_state, t_cross*2, 'black')
    # # ax3.legend(['normalized by position', 'normalized by entire state', 'Lyapunov orbit'])
    # ax3.set_title('velocity space')
    
    # for i in range(len(stableIC1)):
    #     vec = normalize(stvecs2[i][3::])*0.001
    #     ax3.plot([states2[i][3], states2[i][3]+vec[0]], [states2[i][4], states2[i][4]+vec[1]], c='blue')
        
    
    
    
    
    
    
    















