# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:04:27 2023
@author: samantha ramsey

functions to use in multiprocessing integration of lists of initial states
useful for manifold propagation
"""

import pickle
from scipy.integrate import solve_ivp
import numpy as np
import sys


# load in constants
pickle_off = open("general_params.pickle", 'rb')
general_params = pickle.load(pickle_off)
# system specific values
pickle_off = open("system_data.pickle", 'rb')
system_data = pickle.load(pickle_off)


# append CR3BP folder to path so we can call functions from it
path2CR3BP = general_params['path2CR3BP']
sys.path.append(path2CR3BP)

from CR3BP_equations import EOM_2D
from tools import normalize



# define constants
tol = general_params['tol']

# start and end times for propagation incase no event detected
t0 = general_params['t0']
tf = general_params['tf']

# system mass parameter
mu = system_data['mu']


'''
-------------------------------------------------------------------------------

event definitions

could also define the direction of crossing (event.direction = +/- 1)
if positive event will trigger when going from negative to positive
vice-versa if event is set to negative
if undefined the default is 0 and event will trigger in either direction

-------------------------------------------------------------------------------
'''

def cross_primary(t, state):
    ''' trajectory crosses the x-position of the larger primary '''
    return abs(state[0] - mu)
cross_primary.terminal = True

def cross_secondary(t, state):
    ''' trajectory crosses the x-position of the smaller primary '''
    return state[0] - (1-mu)
cross_secondary.terminal = True

def cross_xaxis(t, state):
    ''' trajectory crosses the x-axis when y=0 '''
    return state[1] - 1e-100
cross_xaxis.terminal = True

def cross_yaxis(t, state):
    ''' trajectory crosses the y-axis when x=0 '''
    return state[0] - 1e-100
cross_yaxis.terminal = True

def apse(t, state):
    ''' trajectory is at an apse when the dot product of r and v is 0 '''
    re = np.array([1-mu, 0.0])
    r  = normalize(state[0:2]-re)
    v  = normalize(state[2::])
    return np.dot(r, v)
apse.terminal = True


'''
-------------------------------------------------------------------------------

integration functions to pass into multiprocessing
stable manifolds propagated in negative time, unstable in positive time

-------------------------------------------------------------------------------
'''

# ---------------------------------- stable manifolds
def stable_to_secondary(IC):
    # integrate in negative time until the trajectory crosses the smaller primary
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -tf), IC,
                    dense_output=True, events=cross_secondary, rtol=tol, atol=tol)
    return sol

def stable_to_primary(IC):
    # integrate in negative time until the trajectory crosses the larger primary
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -tf), IC,
                    dense_output=True, events=cross_primary, rtol=tol, atol=tol)
    return sol

def stable_to_xaxis(IC):
    # integrate in negative time until the trajectory crosses the xaxis
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -tf), IC,
                    dense_output=True, events=cross_xaxis, rtol=tol, atol=tol)
    return sol

def stable_to_yaxis(IC):
    # integrate in negative time until the trajectory crosses the yaxis
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -tf), IC,
                    dense_output=True, events=cross_yaxis, rtol=tol, atol=tol)
    return sol

def stable_to_apse(IC):
    # integrate in negative time until the trajectory reaches an apse
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -tf), IC,
                    dense_output=True, events=apse, rtol=tol, atol=tol)
    return sol


# ---------------------------------- unstable manifolds
def unstable_to_secondary(IC):
    # integrate in negative time until the trajectory crosses the smaller primary
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tf), IC,
                    dense_output=True, events=cross_secondary, rtol=tol, atol=tol)
    return sol

def unstable_to_primary(IC):
    # integrate in negative time until the trajectory crosses the larger primary
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tf), IC,
                    dense_output=True, events=cross_primary, rtol=tol, atol=tol)
    return sol

def unstable_to_xaxis(IC):
    # integrate in negative time until the trajectory crosses the xaxis
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tf), IC,
                    dense_output=True, events=cross_xaxis, rtol=tol, atol=tol)
    return sol

def unstable_to_yaxis(IC):
    # integrate in negative time until the trajectory crosses the yaxis
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tf), IC,
                    dense_output=True, events=cross_yaxis, rtol=tol, atol=tol)
    return sol

def unstable_to_apse(IC):
    # integrate in negative time until the trajectory reaches an apse
    sol = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tf), IC,
                    dense_output=True, events=apse, rtol=tol, atol=tol)
    return sol





















