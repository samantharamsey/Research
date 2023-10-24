# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:43:55 2023

@author: saman
"""

import numpy as np
import itertools
import pickle
from scipy.integrate import solve_ivp

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
import re
import ast

# append CR3BP folder to path so we can call functions from it
path2CR3BP = general_params['path2CR3BP']
sys.path.append(path2CR3BP)

from CR3BP_equations import EOM_STM, EOM_STM_2D

#https://stackoverflow.com/a/44323021
def DataFrame_str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))




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


def normalize(x):
    ''' normalize a vector into a unit vector '''
    return x/np.linalg.norm(x)


def sort_eigenvalues(eig_vals, eig_vecs, weight_reciprocal=1.0, weight_subsequent=1.0, weight_dot=1.0):
    """
    This algorithm leverages three simple axioms to determine if the eigenvalues
    are in a consistent order.

        1. Pairs of eigenvalues should occur in sequence ( [0,1] or [2,3], etc.) and have
        a product equal to one (reciprocal pairs)

        2. Eigenvalues evolve continuously along the family, thus the changes between subsequent
        eigenvalues should be small

        3. Eigenvectors evolve continuously along the family, thus the dot product between subsequent
        eigenvectors should be small

    Axiom #1 is not always applicable to trajectories but is always applicable to families of periodic
    orbits in the CR3BP

    :param eig_vals: eigenvectors across family
    :param eig_vecs: eigenvalues across family
    :return:
    """
    eig_vals = np.array(eig_vals)
    eig_vecs = np.array(eig_vecs)

    if not len(eig_vals):
        print('No eigenvalues - easy to sort! :P')
        return

    print('Sorting %i Eigenvalues' % len(eig_vals))
    sorted_ixs = []
    sorted_eig_vals = []
    sorted_eig_vecs = []

    # Generate all permutations of the indices 0 through 5
    ix_perms = [p for p in itertools.permutations(list(range(6)))]
    cost = np.zeros(len(ix_perms))

    # Sort the first set of eigenvalues so that reciprocal pairs occur near one another
    for p, permutation in enumerate(ix_perms):
        for i in range(3):
            cost[p] += abs(
                1.0 -
                eig_vals[0][permutation[2 * i]] *
                eig_vals[0][permutation[2 * i + 1]]
            )

    # Find the minimum cost
    min_cost_ix = np.argmin(cost)

    # Sort the eigenvectors and eigenvalues according to the minimum cost permutation
    sorted_ixs.append(ix_perms[min_cost_ix])
    prev_sort_val = [eig_vals[0][i] for i in sorted_ixs[0]]
    prev_sort_vec = [eig_vecs[0][i] for i in sorted_ixs[0]]

    sorted_eig_vals.append(prev_sort_val)
    sorted_eig_vecs.append(prev_sort_vec)

    # Analyze all other eigenvalue sets using axioms while maintaining order of the first set of eigenvalues/vectors
    for s, eig_val in enumerate(eig_vals[1:]):

        dp_err = np.zeros((6, 6))  # Dot Product Error

        for i, prev_eig_vec in enumerate(prev_sort_vec):
            for j in range(0, 6):
                dp_err[i][j] = abs(1.0 - abs(prev_eig_vec.dot(eig_vecs[s][j])))
                # dp_err[j][i] = dp_err[i][j]

        # Reset the costs
        cost = np.zeros(len(ix_perms))

        for p, permutation in enumerate(ix_perms):
            for i, pix in enumerate(permutation):
                # Assign cost based on the dot product between this new arrangement
                # of eigenvectors and the previous arrangement
                cost[p] += dp_err[i, pix] * weight_dot

                # Assign cost based on the distance from the previous sorted eigenvalues
                cost[p] += abs(eig_val[pix] - prev_sort_val[i]) * weight_subsequent

                # Assign cost based on the reciprocal nature of the eigenvalues
                # This test could be removed to sort eigenvalues that don't occur in pairs
                if i % 2 == 1:
                    cost[p] += abs(
                        1.0 - eig_val[permutation[i - 1]] * eig_val[permutation[i]]) * weight_reciprocal

        # Find the minimum cost
        min_cost_ix = np.argmin(cost)

        sorted_ixs.append(ix_perms[min_cost_ix])
        prev_sort_val = [eig_val[i] for i in sorted_ixs[-1]]
        prev_sort_vec = [eig_vecs[s][i] for i in sorted_ixs[-1]]

        sorted_eig_vals.append(prev_sort_val)
        sorted_eig_vecs.append(prev_sort_vec)

    return sorted_eig_vals, sorted_eig_vecs

