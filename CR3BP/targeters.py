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

from CR3BP_equations import EOM, EOM_2D, EOM_STM, EOM_STM_2D, jacobi_constant
from tools import normalize

# define constants
tol = general_params['tol']

# start and end times for propagation incase no event detected
t0 = general_params['t0']
tf = general_params['tf']

# system mass parameter
mu = system_data['mu']


# want to target a perpendicaular crossing at the x-axis
def cross_xaxis_ypos2neg(t, x):
    ''' trajectory crosses the x-axis when y=0 '''
    return x[1] - 1e-100
cross_xaxis_ypos2neg.terminal  = True
cross_xaxis_ypos2neg.direction = -1

# want to target a perpendicaular crossing at the x-axis
def cross_xaxis_yneg2pos(t, x):
    ''' trajectory crosses the x-axis when y=0 '''
    return x[1] + 1e-100
cross_xaxis_yneg2pos.terminal  = True
cross_xaxis_yneg2pos.direction = 1

# want to target a perpendicaular crossing at the x-axis
def cross_xaxis_zneg2pos(t, x):
    ''' trajectory crosses the x-axis when z=0 '''
    z = x[2] + 1e-100
    return z
cross_xaxis_zneg2pos.terminal  = True
cross_xaxis_zneg2pos.direction = 1

# want to target a perpendicaular crossing at the x-axis
def cross_xaxis_zpos2neg(t, x):
    ''' trajectory crosses the x-axis when z=0 '''
    z = x[2] - 1e-100
    return z
cross_xaxis_zpos2neg.terminal  = True
cross_xaxis_zpos2neg.direction = -1


def xyplane_vzfixed(state0, pos2neg=False, return_STM=False):
    '''
    targets x-axis crossing with fixed z-velocity
    '''
    x0, y0, z0, vx0, vy0, vz0 = state0
    
    error = 1
    iteration = 0
    
    while error > tol:
        
        # initialize the STM with identity matrix reshaped to an array
        phi0 = np.identity(6)
        phi0 = phi0.reshape(1, 36)
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # use solve_ivp to integrate over time
        if pos2neg:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        events=cross_xaxis_zpos2neg, rtol=tol, atol=tol)
        else:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        events=cross_xaxis_zneg2pos, rtol=tol, atol=tol)
        
        t_cross = sol.t_events[0]
        phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
            
        statef = phi[0:6]
        
        STM = phi[6::].reshape((6, 6))
        
        # targeting a perpandicular crossing at x-axis: yf = 0, zf=0, xdotf=0
        error_y    = 0 - statef[1]
        error_xdot = 0 - statef[3]
        error = np.array([error_y, error_xdot]).T
        
        # get derivative of state
        dstatef = EOM(0, statef, mu)
        
        # pull elements from dstate
        ydot  = dstatef[1]
        zdot  = dstatef[2]
        xddot = dstatef[3]
        
        # update matrix
        update  = np.array([[STM[1][0] - ( ydot/zdot)*STM[2][0], STM[1][4] - ( ydot/zdot)*STM[2][4]],
                            [STM[3][0] - (xddot/zdot)*STM[2][0], STM[3][4] - (xddot/zdot)*STM[2][4]]])
        
        delta_IC = np.linalg.solve(update, error)
        
        # change to initial x position
        x0  += delta_IC[0]
        # change to initial y velocity
        vy0 += delta_IC[1]
        
        state0 = np.array([x0, y0, z0, vx0, vy0, vz0])
        error = abs(np.linalg.norm(error))
        
        iteration += 1
        if iteration > 100:
            print('iteration max reached, ', error)
            break
    
    if return_STM==True:
        return state0, t_cross[0], STM
    else:
        return state0, t_cross[0]
    
def xyplane_tfixed(state0, P, pos2neg=False, return_STM=False):
    '''
    targets a perpandicular crossing w fixed period
    '''
    
    x0, y0, z0, vx0, vy0, vz0 = state0
    
    error = 1
    iteration = 0
    
    while error > tol:
        
        # initialize the STM with identity matrix reshaped to an array
        phi0 = np.identity(6)
        phi0 = phi0.reshape(1, 36)
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # use solve_ivp to integrate over time
        sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, P), state0, method='LSODA', 
                    dense_output=True, rtol=tol, atol=tol)
            
        phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
            
        statef = phi[0:6]
        STM = phi[6::].reshape((6, 6))
        
        # targeting a perpandicular crossing at x-y plane
        error_y  = 0 - statef[1]
        error_z  = 0 - statef[2]
        error_dx = 0 - statef[3]
        error = np.array([error_y, error_z, error_dx]).T
        
        # update matrix
        update  = np.array([[STM[1][0], STM[1][4], STM[1][5]],
                            [STM[2][0], STM[2][4], STM[2][5]],
                            [STM[3][0], STM[3][4], STM[3][5]]])
        
        delta_IC = np.linalg.solve(update, error)
        
        # change to initial x position
        x0  += delta_IC[0]
        # change to initial y velocity
        vy0 += delta_IC[1]
        # change to initial z velocity
        vz0 += delta_IC[2]
        
        state0 = np.array([x0, y0, z0, vx0, vy0, vz0])
        error = abs(np.linalg.norm(error))
        
        iteration += 1
        if iteration > 15:
            print('iteration max reached, ', error)
            break
    
    if return_STM==True:
        return state0, STM
    else:
        return state0
    
def xyplane_vyfixed(state0, pos2neg=False, return_STM=False):
    '''
    targets x-axis crossing with fixed y-velocity
    '''
    x0, y0, z0, vx0, vy0, vz0 = state0
    
    error = 1
    iteration = 0
    
    while error > tol:
        
        # initialize the STM with identity matrix reshaped to an array
        phi0 = np.identity(6)
        phi0 = phi0.reshape(1, 36)
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # use solve_ivp to integrate over time
        if pos2neg:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_ypos2neg, rtol=tol, atol=tol)
        else:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_yneg2pos, rtol=tol, atol=tol)
        
        t_cross = sol.t_events[0]
        phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
            
        statef = phi[0:6]
        
        STM = phi[6::].reshape((6, 6))
        
        # targeting a perpandicular crossing at x-axis: zf = 0, zf=0, xdotf=0
        error_z    = 0 - statef[2]
        error_xdot = 0 - statef[3]
        error = np.array([error_z, error_xdot]).T
        
        # get derivative of state
        dstatef = EOM(0, statef, mu)
        
        # pull elements from dstate
        ydot  = dstatef[1]
        zdot  = dstatef[2]
        xddot = dstatef[3]
        
        # update matrix
        update  = np.array([[STM[2][0] - ( zdot/ydot)*STM[1][0], STM[2][4] - ( zdot/ydot)*STM[1][4]],
                            [STM[3][0] - (xddot/ydot)*STM[1][0], STM[3][4] - (xddot/ydot)*STM[1][4]]])
        
        delta_IC = np.linalg.solve(update, error)
        
        # change to initial x position
        x0  += delta_IC[0]
        # change to initial y velocity
        vy0 += delta_IC[1]
        
        state0 = np.array([x0, y0, z0, vx0, vy0, vz0])
        error = abs(np.linalg.norm(error))
        
        iteration += 1
        if iteration > 100:
            print('iteration max reached, ', error)
            break
    
    if return_STM==True:
        return state0, t_cross[0], STM
    else:
        return state0, t_cross[0]
    

def xzplane_tfixed(state0, P, return_STM=False):
    '''
    targets a perpandicular crossing w fixed period
    '''
    
    x0, y0, z0, vx0, vy0, vz0 = state0
    
    error = 1
    iteration = 0
    
    while error > tol:
        
        # initialize the STM with identity matrix reshaped to an array
        phi0 = np.identity(6)
        phi0 = phi0.reshape(1, 36)
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # use solve_ivp to integrate over time
        sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, P), state0, method='LSODA', 
                    dense_output=True, rtol=tol, atol=tol)
            
        phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
            
        statef = phi[0:6]
        STM = phi[6::].reshape((6, 6))
        
        # targeting a perpandicular crossing at x-y plane
        error_y  = 0 - statef[1]
        error_dx = 0 - statef[3]
        error_dz = 0 - statef[5]
        error = np.array([error_y, error_dx, error_dz]).T
        
        # update matrix
        update  = np.array([[STM[1][0], STM[1][2], STM[1][4]],
                            [STM[3][0], STM[3][2], STM[3][4]],
                            [STM[5][0], STM[5][2], STM[5][4]]])
        
        delta_IC = np.linalg.solve(update, error)
        
        # change to initial x position
        x0  += delta_IC[0]
        # change to initial z position
        z0  += delta_IC[1]
        # change to initial y velocity
        vy0 += delta_IC[2]
        
        state0 = np.array([x0, y0, z0, vx0, vy0, vz0])
        error = abs(np.linalg.norm(error))
        
        iteration += 1
        if iteration > 50:
            print('iteration max reached, ', error)
            break
    
    if return_STM==True:
        return state0, STM
    else:
        return state0
    
def xzplane_xfixed(state0, pos2neg=False, return_STM=False):
    '''
    targets x-z plane crossing with fixed x-position
    '''
    x0, y0, z0, vx0, vy0, vz0 = state0
    
    error = 1
    iteration = 0
    
    while error > tol:
        
        # initialize the STM with identity matrix reshaped to an array
        phi0 = np.identity(6)
        phi0 = phi0.reshape(1, 36)
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # use solve_ivp to integrate over time
        if pos2neg:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_ypos2neg, rtol=tol, atol=tol)
        else:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_yneg2pos, rtol=tol, atol=tol)
        
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
        z0  += delta_IC[0]
        # change to initial y velocity
        vy0 += delta_IC[1]
        
        state0 = np.array([x0, y0, z0, vx0, vy0, vz0])
        error = abs(np.linalg.norm(error))
        # print(state0)
        
        iteration += 1
        if iteration > 100:
            print('iteration max reached, ', error)
            break
    
    if return_STM==True:
        return state0, t_cross[0], STM
    else:
        return state0, t_cross[0]
    
def xzplane_zfixed(state0, pos2neg=False, return_STM=False):
    '''
    targets x-z plane crossing with fixed z-position
    '''
    x0, y0, z0, vx0, vy0, vz0 = state0
    
    error = 1
    iteration = 0
    
    while error > tol:
        
        # initialize the STM with identity matrix reshaped to an array
        phi0 = np.identity(6)
        phi0 = phi0.reshape(1, 36)
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # use solve_ivp to integrate over time
        if pos2neg:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_ypos2neg, rtol=tol, atol=tol)
        else:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_yneg2pos, rtol=tol, atol=tol)
        
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
        update  = np.array([[STM[3][0] - (xddot/ydot)*STM[1][0], STM[3][4] - (xddot/ydot)*STM[1][4]],
                            [STM[5][0] - (zddot/ydot)*STM[1][0], STM[5][4] - (zddot/ydot)*STM[1][4]]])
        # print(update)
        # print(error)
        delta_IC = np.linalg.solve(update, error)
        
        # change to initial x position
        x0  += delta_IC[0]
        # change to initial y velocity
        vy0 += delta_IC[1]
        
        state0 = np.array([x0, y0, z0, vx0, vy0, vz0])
        error = abs(np.linalg.norm(error))
        # print(state0)
        
        iteration += 1
        if iteration > 100:
            print('iteration max reached, ', error)
            break
    
    if return_STM==True:
        return state0, t_cross[0], STM
    else:
        return state0, t_cross[0]
    

def xzplane_vyfixed(state0, pos2neg=False, return_STM=False):
    '''
    targets x-z plane crossing with fixed x-position
    '''
    x0, y0, z0, vx0, vy0, vz0 = state0
    
    error = 1
    iteration = 0
    
    while error > tol:
        
        # initialize the STM with identity matrix reshaped to an array
        phi0 = np.identity(6)
        phi0 = phi0.reshape(1, 36)
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # use solve_ivp to integrate over time
        if pos2neg:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_ypos2neg, rtol=tol, atol=tol)
        else:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_yneg2pos, rtol=tol, atol=tol)
        
        t_cross = sol.t_events[0]
        phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
            
        statef = phi[0:6]
        
        STM = phi[6::].reshape((6, 6))
        
        # targeting a perpandicular crossing at x-z plane and x-y planes
        error_vx = 0 - statef[3]
        error_vz = 0 - statef[5]
        error    = np.array([error_vx, error_vz]).T
        
        # get derivative of state
        dstatef = EOM(0, statef, mu)
        
        # pull elements from dstate
        ydot  = dstatef[1]
        xddot = dstatef[3]
        zddot = dstatef[5]
        
        # update matrix
        update  = np.array([[STM[3][0] - (xddot/ydot)*STM[1][0], STM[3][5] - (xddot/ydot)*STM[1][5]],
                            [STM[5][0] - (zddot/ydot)*STM[1][0], STM[5][5] - (zddot/ydot)*STM[1][5]]])

        delta_IC = np.linalg.solve(update, error)
        
        # change to initial x position
        x0 += delta_IC[0]
        # change to initial y velocity
        vz0 += delta_IC[1]
        
        state0 = np.array([x0, y0, z0, vx0, vy0, vz0])
        error = abs(np.linalg.norm(error))
        
        iteration += 1
        if iteration > 100:
            print('iteration max reached, ', error)
            break
    
    if return_STM==True:
        return state0, t_cross[0], STM
    else:
        return state0, t_cross[0]
    
def xzplane_vzfixed(state0, pos2neg=False, return_STM=False):
    '''
    targets x-z plane crossing with fixed x-position
    '''
    x0, y0, z0, vx0, vy0, vz0 = state0
    
    error = 1
    iteration = 0
    
    while error > tol:
        
        # initialize the STM with identity matrix reshaped to an array
        phi0 = np.identity(6)
        phi0 = phi0.reshape(1, 36)
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # use solve_ivp to integrate over time
        if pos2neg:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_ypos2neg, rtol=tol, atol=tol)
        else:
            sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state0, method='LSODA', 
                        dense_output=True, events=cross_xaxis_yneg2pos, rtol=tol, atol=tol)
        
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
        update  = np.array([[STM[3][0] - (xddot/ydot)*STM[1][0], STM[3][4] - (xddot/ydot)*STM[1][4]],
                            [STM[5][0] - (zddot/ydot)*STM[1][0], STM[5][4] - (zddot/ydot)*STM[1][4]]])
        # print(update)
        # print(error)
        delta_IC = np.linalg.solve(update, error)
        
        # change to initial x position
        x0  += delta_IC[0]
        # change to initial y velocity
        vy0 += delta_IC[1]
        
        state0 = np.array([x0, y0, z0, vx0, vy0, vz0])
        error = abs(np.linalg.norm(error))
        
        iteration += 1
        if iteration > 100:
            print('iteration max reached, ', error)
            break
    
    if return_STM==True:
        return state0, t_cross[0], STM
    else:
        return state0, t_cross[0]
    


def xzplane_JCfixed(state0, P, desiredJC, attenuation=1, return_STM=False):
    '''
    targets x-z plane crossing with fixed JC
    '''
    
    # unpack state vector
    x0, y0, z0, dx0, dy0, dz0 = state0
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(6)
    phi0 = phi0.reshape(1, 36)
    
    error = 1
    iteration = 0
    
    # -------------------------------------------------------------------------
    # free variable vector
    # -------------------------------------------------------------------------
    
    # we can choose to alter the initial x z position, y velocity, and period
    X = np.array([x0, z0, dy0, P])
    
    while error > tol:
        
        # add STM to state
        state0 = np.concatenate([state0, phi0[0]])
        
        # get derivative of initial state
        dstate0 = EOM(0, state0, mu)
        dx0, dy0, dz0, ddx0, ddy0, ddz0 = dstate0
        
        # ---------------------------------------------------------------------
        # constraint vector
        # ---------------------------------------------------------------------
        
        # use solve_ivp to integrate over time
        sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, X[-1]), state0, 
                    dense_output=True, rtol=tol, atol=tol, method='LSODA')
        
        # get state and STM at final point in integration
        phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
        STM = phi[6::].reshape((6, 6))
        statef = phi[0:6]
        
        # get derivative of final state
        dstatef = EOM(0, statef, mu)
        xf, yf, zf = statef[0:3]
        dxf, dyf, dzf, ddxf, ddyf, ddzf = dstatef
        
        # want to match the desired JC
        currentJC = jacobi_constant(mu, state0[0:6])
        JCdiff = desiredJC - currentJC
        
        # want perpendicular crossing at x-z plane, so y, dx, dz must be 0
        F = np.array([yf, dxf, dzf, JCdiff])
        
        # ---------------------------------------------------------------------
        # jacobian - partial derivative of F wrt X
        # ---------------------------------------------------------------------
        
        # define the magnitude of the vector to each primary
        d = np.sqrt((x0     + mu)**2 + y0**2 + z0**2)
        r = np.sqrt((x0 - 1 + mu)**2 + y0**2 + z0**2)
        
        # partial of difference in Jacobi Constant
        dJCdx  = -2*((1 - mu)*(x0 + mu)/d**3 + mu*(x0 - 1 + mu)/r**3 - x0)
        dJCdz  = -2*((1 - mu)* z0      /d**3 + mu* z0          /r**3 - x0)
        dJCdyd = -2*dy0
        dJCdt  =  0
        
        # partial derivatives of constraints wrt the free variables
        dFdX = np.array([[STM[1][0], STM[1][2], STM[1][4],   dyf],
                         [STM[3][0], STM[3][2], STM[3][4],  ddxf],
                         [STM[5][0], STM[5][2], STM[5][4],  ddzf],
                         [    dJCdx,     dJCdz,    dJCdyd, dJCdt]])
        
        # update to free variable vector
        update = np.linalg.pinv(dFdX)@F
        
        # change to free variable vector
        X = X - attenuation*update
        
        # updated initial state
        state0 = np.array([X[0], y0, X[1], dx0, X[2], dz0])
        
        error = abs(np.linalg.norm(F))
        
        iteration += 1
        if iteration > 20:
            print('iteration max reached, ', error)
            break
    
    
    if return_STM==True:
        return state0, X[-1], STM
    else:
        return state0, X[-1]



# def xzplane_xfixed_2D(state0):
#     '''
#     targets x-z plane crossing with fixed x-position
#     Parameters
#     ----------
#     state0 : TYPE
#         DESCRIPTION.
#     mu : TYPE
#         DESCRIPTION.
#     event : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     state0 : TYPE
#         DESCRIPTION.
#     TYPE
#         DESCRIPTION.
#     '''
#     xi0, eta0, v_xi0, v_eta0 = state0
    
#     error = 1
#     iteration = 0
    
#     while error > tol:
    
#         # initialize the STM with identity matrix reshaped to an array
#         phi0 = np.identity(4)
#         phi0 = phi0.reshape(1, 16)
        
#         # add STM to state
#         state0 = np.concatenate([state0, phi0[0]])
        
#         # use solve_ivp to integrate over time
#         sol = solve_ivp(lambda t, y: EOM_STM_2D(t, y, mu), (t0, tf), state0, 
#                         dense_output=True, events=cross_xaxis, rtol=tol, atol=tol)
        
#         t_cross = sol.t_events[0]
#         phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
            
#         statef = phi[0:4]
        
#         STM = phi[4::].reshape((4, 4))
        
#         # targeting a perpandicular crossing at x-axis: xdotf = 0, y=0
#         error = 0 - statef[2] 
#         dstatef = EOM_2D(0, statef, mu)
#         ydot = dstatef[1]
#         xddot = dstatef[2]
        
#         update = STM[2][3] - STM[1][3]*(xddot/ydot)
#         del_ydot = error/update
#         v_eta0 += del_ydot
        
#         state0 = np.array([xi0, eta0, v_xi0, v_eta0])
#         error = abs(error)
        
#         iteration += 1
#         if iteration > 20:
#             print('iteration max reached')
#             break
    
#     return state0, t_cross[0]

