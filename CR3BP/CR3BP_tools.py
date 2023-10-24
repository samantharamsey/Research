# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:43:55 2023

@author: saman
"""

import numpy as np
import itertools
import pickle
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# load in constants
pickle_off = open("general_params.pickle", 'rb')
general_params = pickle.load(pickle_off)
# system specific values
pickle_off = open("system_data.pickle", 'rb')
system_data = pickle.load(pickle_off)
    
# define constants
tol = general_params['tol']

# start and end times for propagation incase no event detected
t0 = general_params['t0']
tf = general_params['tf']

# system mass parameter
mu = system_data['mu']
# step off distance
stepoff = system_data['stepoff']

import sys
# append CR3BP folder to path so we can call functions from it
path2CR3BP = general_params['path2CR3BP']
sys.path.append(path2CR3BP)

from CR3BP_equations import EOM_STM, EOM_STM_2D



def make_family_data(states, times, JCs, MonMs, sorted_vals, sorted_vecs,
                     plot_results=False):
    '''
    make a DataFrame out of the data for a family of periodic orbits
        calculates the stability indicies, time constants, Lyapunov exponents,
        and alpha and beta terms for each orbit in the family
    '''
    # empty lists to store calculated data
    stab_idxs, time_consts = [], []
    alphas, betas = [], []
    Lyap_exp  = []
    
    # now that we have all the monodromy matricies we can iterate through them
    for i in range(len(MonMs)):
        
        # period of current orbit
        t = times[i]
        
        # monodromy matrix of current orbit
        monod = MonMs[i]
        
        # compute stability indicies (zimovan 2021 pg 68)
        abs_val = [abs(v) for  v in sorted_vals[i]]
        stab  = np.array([(1/2)*(j + 1/j) for j in abs_val])
        
        # compute time constant from max eigenvalue (zimovan 2021 pg 71)
        max_val = max([abs(sorted_vals[i][j]) for j in range(6)])
        nat_val = np.log(max_val)
        t_cons  = 1/(abs(np.real(nat_val))*t)
        
        # Lyapunov exponents lambda = e^(alpha*t)
        # e^x=y ---> ln(y)=x
        exps = [np.log(v)/t for  v in sorted_vals[i]]
        
        # alpha and beta for broucke stability diagram
        alpha = 2 - np.trace(monod)
        beta  = (1/2)*(alpha**2 + 2 - np.trace(monod@monod))
        
        # add all values to their respective lists
        stab_idxs.append(stab)
        time_consts.append(t_cons)
        alphas.append(alpha)
        betas.append(beta)
        Lyap_exp.append(exps)
    
    # pass corrected initial conditions, period, JC, and other values to a DF
    ICData = pd.DataFrame({'initial condition': states, 'P': times, 'JC': JCs, 
                            'time constant': time_consts,
                            'alpha': alphas, 'beta': betas})
    
    # eigenvales and vectors
    valData = pd.DataFrame(sorted_vals, columns=['val 1', 'val 2', 'val 3',
                                                  'val 4', 'val 5', 'val 6'])
    
    vecData = pd.DataFrame(sorted_vecs, columns=['vec1', 'vec2', 'vec3',
                                                  'vec4', 'vec5', 'vec6'])
    
    # stability indicies 
    stbData = pd.DataFrame(stab_idxs, columns=['stb1', 'stb2', 'stb3',
                                                'stb4', 'stb5', 'stb6'])
    
    # Lyapunov exponents
    expData = pd.DataFrame(Lyap_exp, columns=['Lyap exp 1', 'Lyap exp 2', 'Lyap exp 3',
                                              'Lyap exp 4', 'Lyap exp 5', 'Lyap exp 6'])
    
    # add monodromy matrix to eigenvector data
    vecData['monodromy matrix'] = pd.Series(MonMs)
    
    FamilyData = ICData.join([stbData, expData, valData, vecData])

    
    if plot_results==True:
        # plot time constant
        fig1, ax1 = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 9),)
        fig1.set_tight_layout(True)
        
        # plot title
        ax1[0].set_title('Family Characteristics')
        
        # half period
        ax1[0].plot(JCs, times, c='magenta')
        
        # time constant
        ax1[1].plot(JCs, FamilyData['time constant'], c='magenta')
        ax1[1].set_ylim(0, 2)
        
        # stabillity index
        ax1[2].plot(JCs, FamilyData['stb1'], c='magenta')
        ax1[2].plot(JCs, FamilyData['stb2'], c='magenta')
        ax1[2].plot(JCs, FamilyData['stb3'], c='magenta')
        ax1[2].plot(JCs, FamilyData['stb4'], c='magenta')
        ax1[2].plot(JCs, FamilyData['stb5'], c='magenta')
        ax1[2].plot(JCs, FamilyData['stb6'], c='magenta')
        
        # Lyapunov exponents
        ax1[3].plot(JCs, FamilyData['Lyap exp 1'], c='magenta')
        ax1[3].plot(JCs, FamilyData['Lyap exp 2'], c='magenta')
        ax1[3].plot(JCs, FamilyData['Lyap exp 3'], c='magenta')
        ax1[3].plot(JCs, FamilyData['Lyap exp 4'], c='magenta')
        ax1[3].plot(JCs, FamilyData['Lyap exp 5'], c='magenta')
        ax1[3].plot(JCs, FamilyData['Lyap exp 6'], c='magenta')
        
        # grid lines
        ax1[0].grid(alpha=0.2)
        ax1[1].grid(alpha=0.2)
        ax1[2].grid(alpha=0.2)
        ax1[3].grid(alpha=0.2)
        
        # label for each y axis
        ax1[0].set_ylabel('Half Period')
        ax1[1].set_ylabel('Time Constant')
        ax1[2].set_ylabel('Stability Index')
        ax1[3].set_ylabel('Lyapunov Exponents')
        
        # only want x label on bottom axis
        ax1[3].set_xlabel('Jacobi Constant')
    
    return FamilyData


def get_manifold_ICs(mu, orbitIC, P, stepoff, num_manifolds, full_state=False, 
                    positive_dir=True, stable=True, tau_alpha=None,
                    return_eigvecs=False, return_STMs=False, return_fixedpoints=False):
    
    # need set of discreetized points around orbit
    if tau_alpha==None:
        # equally spaced in time for total of num_manifolds points
        tpoints = np.linspace(t0, P, num_manifolds, endpoint=False)
    else:
        # want to get conditions specifically at tau-alpha
        tpoints = np.array([t0, tau_alpha, P])
        # redefine num_manifolds just incase it was passed into function wrong
        num_manifolds = 3
    
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
    # list of stable and unstable eigenvectors
    stvecs, unvecs = [], []
    
    for i, STM in enumerate(STMs):
        
        # make a copy of the state at the current point in the orbit
        state = np.copy(states[i])
        
        # use STM to transition the eigenvectors or the monodromy matrix
        st_vec = STM@stble_vec
        un_vec = STM@unstb_vec
        stvecs.append(st_vec)
        unvecs.append(un_vec)
        
        # normalize by full state or position
        if full_state:
            normst = np.linalg.norm(st_vec)
            normun = np.linalg.norm(un_vec)
        else:
            normst = np.linalg.norm(st_vec[0:3])
            normun = np.linalg.norm(un_vec[0:3])
            
        # perturbation from orbit onto stable/unstable eigenvector
        if stable:
            pert = stepoff*(st_vec/normst)
        else:
            pert = stepoff*(un_vec/normun)
        
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
    
    returns = [manifoldIC, tpoints]
    
    # add in optional returns
    if return_eigvecs==True:
        returns.append(stvecs)
        returns.append(unvecs)
    if return_STMs==True:
        returns.append(STMs)
    if return_fixedpoints==True:
        returns.append(states)
        
    return returns

def get_manifold_ICs_2D(mu, orbitIC, P, stepoff, num_manifolds, full_state=True, 
                    positive_dir=True, stable=True, tau_alpha=None,
                    return_eigvecs=False, return_STMs=False, return_fixedpoints=False):
    
    if tau_alpha==None:
        # need set of discreetized points around orbit
        tpoints = np.linspace(t0, P, num_manifolds, endpoint=False)
        
    else:
        tpoints = np.array([t0, tau_alpha, P])
    
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
    # list of stable and unstable eigenvectors
    stvecs, unvecs = [], []
    
    for i, STM in enumerate(STMs):
        
        # make a copy of the state at the current point in the orbit
        state = np.copy(states[i])
        
        # floquet theory to transition the eigenvectors
        st_vec = STM@stble_vec
        un_vec = STM@unstb_vec
        stvecs.append(st_vec)
        unvecs.append(un_vec)
        
        # normalize by full state or position
        if full_state:
            normst = np.linalg.norm(st_vec)
            normun = np.linalg.norm(un_vec)
        else:
            normst = np.linalg.norm(st_vec[0:2])
            normun = np.linalg.norm(un_vec[0:2])
        
        # perturbation from orbit onto stable/unstable eigenvector
        if stable:
            pert = stepoff*(st_vec/normst)
        else:
            pert = stepoff*(un_vec/normun)
        
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
    
    returns = [manifoldIC, tpoints]
    
    # add in optional returns
    if return_eigvecs==True:
        returns.append(stvecs)
        returns.append(unvecs)
    if return_STMs==True:
        returns.append(STMs)
    if return_fixedpoints==True:
        returns.append(states)
        
    return returns


