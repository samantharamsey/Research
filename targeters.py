# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 18:41:52 2023

@author: sam
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

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
from tools import normalize


# define constants
tol = general_params['tol']

# start and end times for propagation incase no event detected
t0 = general_params['t0']
tf = general_params['tf']

# system mass parameter
mu = system_data['mu']

# want to target a perpendicaular crossing at the x-axis
def cross_xaxis(t, x):
    ''' trajectory crosses the x-axis when y= 0 '''
    return x[1] + 1e-100
cross_xaxis.terminal = True
cross_xaxis.direction = 1


def xzplane_xfixed(state0):
    '''
    targets x-z plane crossing with fixed x-position
    Parameters
    ----------
    state0 : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    event : TYPE
        DESCRIPTION.

    Returns
    -------
    state0 : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    '''
    xi0, eta0, z, v_xi0, v_eta0, vz = state0
    
    error = 1
    iteration = 0
    
    while error > tol:
    
        # initialize the STM with identity matrix reshaped to an array
        phi0 = np.identity(6)
        phi0 = phi0.reshape(1, 36)
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # use solve_ivp to integrate over time
        sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, 
                        dense_output=True, events=cross_xaxis, rtol=tol, atol=tol)
        
        t_cross = sol.t_events[0]
        phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
            
        statef = phi[0:6]
        
        STM = phi[6::].reshape((6, 6))
        
        # targeting a perpandicular crossing at x-z plane: xdotf = 0, zdotf=0
        error_xdot = 0 - statef[3]
        error_zdot = 0 - statef[5]
        error = np.array([error_xdot, error_zdot]).T
        
        # get derivative of state
        dstatef = EOM(0, statef, mu)
        # pull elements from dstate
        ydot  = dstatef[1]
        xddot = dstatef[3]
        zddot = dstatef[5]
        
        # update matrix
        update  = np.array([[STM[3][2] - (xddot/ydot)*STM[1][2], STM[3][4] - (xddot/ydot)*STM[1][4]],
                            [STM[5][2] - (zddot/ydot)*STM[1][2], STM[5][4] - (zddot/ydot)*STM[1][4]]])
        # print(update)
        # print(error)
        delta_IC = np.linalg.solve(update, error)
        
        # change to initial z position
        z      += delta_IC[0]
        # change to initial y velocity
        v_eta0 += delta_IC[1]
        
        state0 = np.array([xi0, eta0, z, v_xi0, v_eta0, vz])
        error = abs(np.linalg.norm(error))
        # print(state0)
        
        iteration += 1
        if iteration > 100:
            print('iteration max reached, ', error)
            break
    
    return state0, t_cross[0]