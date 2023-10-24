# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 22:10:23 2023

@author: sam
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

from CR3BP_equations import EOM, EOM_STM, libsolver, jacobi_constant, EOM_2D, EOM_STM_2D
from tools import normalize, set_axes_equal, get_manifold_ICs, get_manifold_ICs_2D
from targeters import xzplane_xfixed, xzplane_JCfixed
import multiproc_integration as multi

def cross_secondary_pos(t, state):
    ''' trajectory crosses the x-position of the smaller primary '''
    return state[0] - (1-mu)
cross_secondary_pos.terminal  = True
cross_secondary_pos.direction =  1

def cross_secondary_neg(t, state):
    ''' trajectory crosses the x-position of the smaller primary '''
    return state[0] - (1-mu)
cross_secondary_neg.terminal  = True
cross_secondary_neg.direction = -1


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
    
def plot_orbit_2D(ax, orbitIC, P, color, xaxis_symmetry=False):
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    ax.plot(sol.y[0],  sol.y[1], c=color)
    if xaxis_symmetry==True:
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
    
    for i in range(len(states)):
        ax.scatter(states[i][p1], states[i][p2], c=color, s=0.5)
        # ax.scatter(i, states[i][p2], c=color, s=0.5)
    
def find_middle(lst):
    
    # Get the length of the list
    length = len(lst)
    
    # Check if the length is odd
    if length % 2 != 0:
        middle_index = length // 2
        return lst[middle_index]

    # If the length is even
    first_middle_index = length // 2 - 1
    return lst[first_middle_index]
        
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
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(6)
    phi0 = phi0.reshape(1, 36)
    
    # determine the locations of the libration points
    xLib, yLib = libsolver(mu, tol)

    # initial conditions for the Lyapunov orbit at L1
    LyapIC_L1 = np.array([0.8729, 0.0, 0.0, 0.0, -0.2482, 0.0])
    LyapIC_L1, P_L1 = xzplane_xfixed(LyapIC_L1)
    JC_L1 = jacobi_constant(mu, LyapIC_L1)
    
    # initial conditions for the Lyapunov orbit at L2
    LyapIC_L2 = np.array([1.1843, 0.0, 0.0, 0.0, -0.1818, 0.0])
    LyapIC_L2, P_L2 = xzplane_xfixed(LyapIC_L2)
    JC_L2 = jacobi_constant(mu, LyapIC_L2)
    
    
    # 2D plot formatting
    fig1, ax1 = plt.subplots()
    
    
    # plot the two orbits
    plot_orbit_2D(ax1, LyapIC_L1, P_L1*2, 'magenta', xaxis_symmetry=False)
    plot_orbit_2D(ax1, LyapIC_L2, P_L2*2, 'blue', xaxis_symmetry=False)

    
    #%%
    # target an L2 orbit with exactly the same JC as the L1 orbit
    LyapIC_L2, P_L2 = xzplane_JCfixed(LyapIC_L2, P_L2, JC_L1)
    JC_L2 = jacobi_constant(mu, LyapIC_L2)
    
    # plot orbit with corrected JC
    plot_orbit_2D(ax1, LyapIC_L2, P_L2, 'cyan', xaxis_symmetry=True)
    
    ax1.legend(['$L_1$ Lyapunov JC = 3.144098', '$L_2$ Lyapunov JC = 3.144677',
                '$L_2$ Lyapunov JC = 3.144098'])
    
    # add the moon
    rmoon = system_data['r moon']
    moon = plt.Circle((1-mu, 0), rmoon, color='gray', zorder=5)
    ax1.add_patch(moon)
    
    # add the libration points
    ax1.scatter(xLib[0], yLib[0], marker='*', color='yellow', zorder=5)
    ax1.scatter(xLib[1], yLib[1], marker='*', color='yellow', zorder=5)
    
    
    #%%
    
    # 2D plot formatting
    fig1, ax1 = plt.subplots()
    
    # determine total number of processing cores on machine
    ncores = int(mp.cpu_count()/2)
    
    # multiprocess to propagate manifolds
    p = mp.Pool(ncores)
    
    
    # get the unstable manifolds departing the L1 Lyapunov
    num_manifolds = 200
    unstableIC, unTpoints = get_manifold_ICs(mu, LyapIC_L1, 2*P_L1, stepoff, num_manifolds, 
                                positive_dir=True, stable=False)
    
    # integrate stable manifolds to Moon and get final state at Lunar crossing
    integ_states_to_moon = p.map(multi.unstable_to_secondary_pos, unstableIC)
    # index state portion of solve_ivp solution
    unstable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # index time portion of solve_ivp solution
    unstable_times_to_moon = [integ_states_to_moon[i].t[-1] for i in range(len(integ_states_to_moon))]
    # plot and get final state at Lunar crossing
    unstable_states_at_Moon = plot_manifolds_2D(ax1, unstable_states_to_moon, 'red')
    
    # # integrate stable manifolds to Moon and get final state at Lunar crossing
    # integ_states_to_moon = p.map(multi.unstable_to_xaxis, unstable_states_at_Moon)
    # # index state portion of solve_ivp solution
    # unstable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # # index time portion of solve_ivp solution
    # unstable_times_to_moon = [integ_states_to_moon[i].t[-1] for i in range(len(integ_states_to_moon))]
    # # plot and get final state at Lunar crossing
    # unstable_states_at_Moon = plot_manifolds_2D(ax1, unstable_states_to_moon, 'red')
    
    # # integrate stable manifolds to Moon and get final state at Lunar crossing
    # integ_states_to_moon = p.map(multi.unstable_to_secondary, unstable_states_at_Moon)
    # # index state portion of solve_ivp solution
    # unstable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # # index time portion of solve_ivp solution
    # unstable_times_to_moon = [integ_states_to_moon[i].t[-1] for i in range(len(integ_states_to_moon))]
    # # plot and get final state at Lunar crossing
    # unstable_states_at_Moon = plot_manifolds_2D(ax1, unstable_states_to_moon, 'red')
    
    
    # get the stable manifolds arriving at the L2 Lyapunov
    stableIC, stTpoints = get_manifold_ICs(mu, LyapIC_L2, 2*P_L2, stepoff, num_manifolds, 
                                positive_dir=False, stable=True)
    
    # integrate stable manifolds to Moon and get final state at Lunar crossing
    integ_states_to_moon = p.map(multi.stable_to_secondary_neg, stableIC)
    # index state portion of solve_ivp solution
    stable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # index time portion of solve_ivp solution
    stable_times_to_moon = [integ_states_to_moon[i].t[-1] for i in range(len(integ_states_to_moon))]
    # plot and get final state at Lunar crossing
    stable_states_at_Moon = plot_manifolds_2D(ax1, stable_states_to_moon, 'blue')
    
    # # integrate stable manifolds to Moon and get final state at Lunar crossing
    # integ_states_to_moon = p.map(multi.stable_to_xaxis, stable_states_at_Moon)
    # # index state portion of solve_ivp solution
    # stable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # # index time portion of solve_ivp solution
    # stable_times_to_moon = [integ_states_to_moon[i].t[-1] for i in range(len(integ_states_to_moon))]
    # # plot and get final state at Lunar crossing
    # stable_states_at_Moon = plot_manifolds_2D(ax1, stable_states_to_moon, 'blue')
    
    # # integrate stable manifolds to Moon and get final state at Lunar crossing
    # integ_states_to_moon = p.map(multi.stable_to_secondary, stable_states_at_Moon)
    # # index state portion of solve_ivp solution
    # stable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # # index time portion of solve_ivp solution
    # stable_times_to_moon = [integ_states_to_moon[i].t[-1] for i in range(len(integ_states_to_moon))]
    # # plot and get final state at Lunar crossing
    # stable_states_at_Moon = plot_manifolds_2D(ax1, stable_states_to_moon, 'blue')
    

    # plot the two orbits
    plot_orbit_2D(ax1, LyapIC_L1, P_L1, 'black', xaxis_symmetry=True)
    plot_orbit_2D(ax1, LyapIC_L2, P_L2, 'black', xaxis_symmetry=True)
    
    # add the moon
    rmoon = system_data['r moon']
    moon = plt.Circle((1-mu, 0), rmoon, color='gray', zorder=5)
    ax1.add_patch(moon)
    
    # add the libration points
    ax1.scatter(xLib[0], yLib[0], marker='*', color='yellow', zorder=5)
    ax1.scatter(xLib[1], yLib[1], marker='*', color='yellow', zorder=5)
    
    #%%
    fig2, ax2 = plt.subplots()
    ax2.set_title('Poincare Map')
    # ax2.add_patch(moon)
    poincare_section(ax2,   stable_states_at_Moon, 1, 4, 'blue')
    poincare_section(ax2, unstable_states_at_Moon, 1, 4, 'red')
    
    #%%
    
    # -------------------------------------------------------------------------
    # initial guess
    # -------------------------------------------------------------------------
    
    # get initial guess for manifold that terminates closest to perpendicular
    idxs = list(range(0, len(unstable_states_at_Moon)**2))
    
    # want to minimize difference between final unstable state 
    #   and initial stable state
    ids = []
    # iterate through num_manifolds twice to get every permutation
    for i in range(num_manifolds):
        for j in range(num_manifolds):
            diff = np.linalg.norm(np.array(unstable_states_at_Moon[i]) - 
                                  np.array(  stable_states_at_Moon[j]))
            ids.append([abs(diff), i, j])
    # sort in ascending order
    idxs.sort(key = lambda x: ids[x][0])
    
    # choose which initial guess index to use
    ind = 0
    # get stable and unstable indicies 
    unidxs = ids[idxs[ind]][1]
    stidxs = ids[idxs[ind]][2]
    
    # initial guess for tau-alpha will be the Tpoint corresponding to the chosen index
    tau_1 = unTpoints[unidxs]
    tau_2 = stTpoints[stidxs]
    
    # initial condition for unstable manifold IC in 2D
    unIC = unstableIC[unidxs]
    unIC = np.hstack((unIC[0:2], unIC[3:5]))
    
    # integrate until the first time it crosses the moon
    sol_Tu1 = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tf), unIC, 
                    dense_output=True, events=cross_secondary_pos, rtol=tol, atol=tol)
    finalTu1 = np.array([sol_Tu1.y[i][-1] for i in range(4)])
    T_u = sol_Tu1.t[-1]
    # # integrate until the second time it crosses the moon
    # sol_Tu2 = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tf), finalTu1, 
    #                 dense_output=True, events=cross_secondary2, rtol=tol, atol=tol)
    # finalTu2 = np.array([sol_Tu2.y[i][-1] for i in range(4)])
    # Tu2 = sol_Tu2.t[-1]
    # # get total time from orbit to second crossing
    # T_u = Tu1 + Tu2
    
    # initial condition for stable manifold IC in 2D
    stIC = stableIC[stidxs]
    stIC = np.hstack((stIC[0:2], stIC[3:5]))
    
    # integrate until the first time it crosses the moon
    sol_Ts1 = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -tf), stIC, 
                    dense_output=True, events=cross_secondary_neg, rtol=tol, atol=tol)
    finalTs1 = np.array([sol_Ts1.y[i][-1] for i in range(4)])
    T_s = sol_Ts1.t[-1]
    
    
    # # integrate until the second time it crosses the moon
    # sol_Ts2 = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tf), finalTs1, 
    #                 dense_output=True, events=cross_secondary1, rtol=tol, atol=tol)
    # finalTs2 = np.array([sol_Ts2.y[i][-1] for i in range(4)])
    # Ts2 = sol_Ts2.t[-1]
    # # get total time from orbit to second crossing
    # T_s = Ts1 + Ts2
    
    
    
    fig3, ax3 = plt.subplots()
    ax3.set_title('Initial Guess for Heteroclinic Tau-Alpha')
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(alpha=0.2)
    ax3.set_xlabel('x (nd)')
    ax3.set_ylabel('y (nd)')
    
    ax3.plot(sol_Tu1.y[0], sol_Tu1.y[1])
    ax3.plot(sol_Ts1.y[0], sol_Ts1.y[1])
    
    
    #%%
    # initial guess for tau-M is the total integration time along manifold
    # T_u = unstable_times_to_moon[unidxs] 
    # T_s =   stable_times_to_moon[stidxs] 
    
    
    #%% CONVERT TO 2 DIMENSIONAL
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(4)
    phi0 = phi0.reshape(1, 16)
    
    # initial conditions for states along the first periodic orbit
    # state1 is the L1 orbit initial state
    state1 = np.hstack((LyapIC_L1[0:2], LyapIC_L1[3:5]))
    
    # state2 is the fixed point propagated from state1 for t=tau_1 before step off
    # get manifold states at current tau-alpha guess
    un_guess = get_manifold_ICs_2D(mu, state1, P_L1*2, stepoff, 3, 
                                positive_dir=True, stable=False, 
                                tau_alpha=tau_1,
                                return_eigvecs=True, return_STMs=True, 
                                return_fixedpoints=True)
    
    unstableIC1, unTpoints1, stvecs1, unvecs1, STMs1, fixedpoints1 = un_guess
    
    # state2 is the fixed point propagated from state1 for t=tau_1 before step off
    state2 = fixedpoints1[1]
    
    # state3 is the initial condition for the unstable manifold after stepping off from state2
    state3 = unstableIC1[1]
    
    # now need conditions for states along the unstable manifold
    # integrate from t0 to T_u
    sol_Tu = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, T_u), state3, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # state5 is the state at the end of the unstable manifold porpagation
    state5 = np.array([sol_Tu.y[i][-1] for i in range(4)])
    
    # propagate backwards from state5 to get state4
    sol4 = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -T_u), state5, 
                    dense_output=True, rtol=tol, atol=tol)
    state4 = np.array([sol4.y[i][-1] for i in range(4)])


    # initial conditions for states along the second periodic orbit
    # state8 is the L2 orbit initial state
    state8 = np.hstack((LyapIC_L2[0:2], LyapIC_L2[3:5]))
    
    # get manifold states at current tau-alpha guess
    st_guess = get_manifold_ICs_2D(mu, state8, P_L2*2, stepoff, 3, 
                                positive_dir=False, stable=True, 
                                tau_alpha=tau_2,
                                return_eigvecs=True, return_STMs=True, 
                                return_fixedpoints=True)
    
    stableIC2, stTpoints2, stvecs2, unvecs2, STMs2, fixedpoints2 = st_guess
    
    # state9 is the fixed point propagated from state12 for t=tau_2 before step off
    state9 = fixedpoints2[1]
    
    # state10 is the initial condition for the stable manifold after stepping off from state13
    state10 = stableIC2[1]
    
    # now need conditions for states along the stable manifold
    # integrate from t0 to T_s
    sol_Ts = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, T_s), state10, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # state6 is the state at the end of the stable manifold porpagation
    state6 = np.array([sol_Ts.y[i][-1] for i in range(4)])
    
    # propagate backwards (in forward time since stable) from state10 to get state11
    sol7 = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -T_s), state6, 
                    dense_output=True, rtol=tol, atol=tol)
    state7 = np.array([sol7.y[i][-1] for i in range(4)])
    
    
    #%%
    
    
    # -------------------------------------------------------------------------
    # free variable vector
    # -------------------------------------------------------------------------
    
    X = np.concatenate((state4, state7, [tau_1], [tau_2],
                        [T_u], [T_s]), axis=0)
    
    error = 1
    Xnew = X
    iterations = 1
    error_history = []
    f1, f2, f3 = [], [], []
    
    while error > 1e-8:
        
        # update to free variable vector
        X = Xnew
        
        # unpack free variable vector
        r4 = X[0:4]
        r7 = X[4:8]
        t1 = X[8]
        t2 = X[9]
        Tu = X[10]
        Ts = X[11]
        
        
        # -------------------------------------------------------------------------
        # constraint vector
        # -------------------------------------------------------------------------
        
        # integrate along first orbit segment
        un_guess = get_manifold_ICs_2D(mu, state1, P_L1*2, stepoff, 3, 
                                    positive_dir=True, stable=False, 
                                    tau_alpha=t1, full_state=True,
                                    return_eigvecs=True, return_STMs=True, 
                                    return_fixedpoints=True)
        unstableIC1, unTpoints1, stvecs1, unvecs1, STMs1, fixedpoints1 = un_guess
        
        # STM and eigenvectors for first orbit (at r2)
        STM2 = STMs1[1]
        uvec = unvecs1[0]
        # state2 is the fixed point propagated from state1 for t=tau_1 before step off
        r2 = fixedpoints1[1]
        # state3 is the initial condition for the unstable manifold after stepping off from state2
        r3 = unstableIC1[1]
        
        # integrate along first unstable manifold segment
        sol4 = solve_ivp(lambda t, y: EOM_STM_2D(t, y, mu), (t0, Tu), 
                         np.concatenate([r4, phi0[0]]), 
                         dense_output=True, rtol=tol, atol=tol)
        sol5 = np.array([sol4.y[i][-1] for i in range(len(sol4.y))])
        r5   = sol5[0:4]
        STM5 = sol5[4::].reshape((4, 4))
        state5 = r5
        
        # integrate along first stable manifold segment
        sol7 = solve_ivp(lambda t, y: EOM_STM_2D(t, y, mu), (t0, Ts), 
                         np.concatenate([r7, phi0[0]]), 
                         dense_output=True, rtol=tol, atol=tol)
        sol6 = np.array([sol7.y[i][-1] for i in range(len(sol7.y))])
        r6   = sol6[0:4]
        STM6 = sol6[4::].reshape((4, 4))
        state6=r6
        
        # integrate along second orbit segment
        st_guess = get_manifold_ICs_2D(mu, state8, P_L2*2, stepoff, 3, 
                                    positive_dir=False, stable=True, 
                                    tau_alpha=t2, full_state=True,
                                    return_eigvecs=True, return_STMs=True, 
                                    return_fixedpoints=True)
        stableIC2, stTpoints2, stvecs2, unvecs2, STMs2, fixedpoints2 = st_guess
        
        # STM and eigevectors for second orbit (at r9)
        STM9 = STMs2[1]
        svec  = stvecs2[0]
        # state9 is the fixed point propagated from state8 for t=tau_2 before step off
        r9 = fixedpoints2[1]
        # state10 is the initial condition for the stable manifold after stepping off from state9
        r10 = stableIC2[1]
        
        
        # constraint vector
        F = np.concatenate((r4-r3, r6-r5, r7-r10), axis=0)
        f1.append(np.linalg.norm(r4 - r3 ))
        f2.append(np.linalg.norm(r6 - r5 ))
        f3.append(np.linalg.norm(r7 - r10))
        
        # -------------------------------------------------------------------------
        # jacobian - partial derivative of F wrt X
        # -------------------------------------------------------------------------
        
        # time derivatives of each state, plus STM for orbit segments
        dr2   = EOM_STM_2D(0, np.concatenate([r2, STM2.reshape(1, 16)[0]]), mu)
        dSTM2 = dr2[4::].reshape((4, 4))
        dr2   = dr2[0:4]
        
        dr5  = EOM_2D(0,  r5, mu)
        dr6  = EOM_2D(0,  r6, mu)
        
        dr9   = EOM_STM_2D(0, np.concatenate([r9, STM9.reshape(1, 16)[0]]), mu)
        dSTM9 = dr9[4::].reshape((4, 4))
        dr9   = dr9[0:4]
        
        # identity matrix
        I4 = np.identity(4)
        # zero matrix
        Z4 = np.zeros(16).reshape(4, 4)
        # zero array
        Z1 = np.zeros(4).reshape(4, 1)
        
        # norm of eigenvector at tau-alpha
        norm2 = np.linalg.norm(STM2@uvec)
        # splitting math into smaller terms to simplify
        term2 = STM2@uvec*(uvec.T@dSTM2.T@ STM2@uvec + 
                           uvec.T@ STM2.T@dSTM2@uvec)
        # full derivative
        dr3dtau = dr2 + stepoff*((dSTM2@uvec)/norm2 - term2/(2*norm2**3))
        
        # norm of eigenvector at tau-alpha
        norm9 = np.linalg.norm(STM9@svec)
        # splitting math into smaller terms to simplify
        term9 = STM9@svec*(svec.T@dSTM9.T@ STM9@svec + 
                             svec.T@ STM9.T@dSTM9@svec)
        # full derivative
        dr10dtau = dr9 + stepoff*((dSTM9@svec)/norm9 - term9/(2*norm9**3))
        
        # form jacobian
        row1 = np.concatenate((I4, Z4, -dr3dtau.reshape(4, 1), Z1, Z1, Z1), 
                               axis=1)
        
        row2 = np.concatenate((-STM5, STM6, Z1, Z1, -dr5.reshape(4, 1), dr6.reshape(4, 1)), 
                               axis=1)
        
        row3 = np.concatenate((Z4, I4, Z1, -dr10dtau.reshape(4, 1), Z1, Z1), 
                               axis=1)
        
        DF = np.vstack((row1, row2, row3))
        
        update = np.linalg.pinv(DF)@F
        # update = DF.T@(DF@DF.T)**(-1)@F
        # update = DF.T@np.linalg.pinv(DF@DF.T)@F
        
        # update the free variable vector
        Xnew = X - update
        iterations += 1
        error = np.linalg.norm(F)
        error_history.append(error)
        
        if iterations > 10:
            print('iteration max reached, error = {}'.format(error))
            break
        
        
        
    
        
        
        
        
    #%%
    # -------------------------------------------------------------------------
    # plot final result
    # -------------------------------------------------------------------------
    
    # unpack free variable vector
    state4 = X[0:4]
    state7 = X[4:8]
    t1 = X[8]
    t2 = X[9]
    Tu = X[10]
    Ts = X[11]
    
    # convert states back to 6D arrays
    state1 = LyapIC_L1
    state4 = np.insert(state4, 2, 0)
    state4 = np.insert(state4, 5, 0)
    state5 = np.insert(state5, 2, 0)
    state5 = np.insert(state5, 5, 0)
    state6 = np.insert(state6, 2, 0)
    state6 = np.insert(state6, 5, 0)
    state7 = np.insert(state7, 2, 0)
    state7 = np.insert(state7, 5, 0)
    state8 = LyapIC_L2
    #%%
    # start homoclinic connection process
    fig3, ax3 = plt.subplots()
    ax3.set_title('Final Result for Heteroclinic Tau-Alpha')
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(alpha=0.2)
    ax3.set_xlabel('x (nd)')
    ax3.set_ylabel('y (nd)')
    
    
    # unstable half
    # integrate from t0 to tau_alpha
    # sol_tau1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, t1), state1, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_tau1.y[0],  sol_tau1.y[1], zorder=3)
    
    # integrate from tau_1 to T_u1
    sol_Tu = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, Tu+abs(Ts)), state4, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_Tu.y[0],  sol_Tu.y[1], zorder=3, c='magenta')
    
    
    # stable half
    
    # integrate from tau_1 to T_u1
    # sol_Ts = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, Ts), state7, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_Ts.y[0],  sol_Ts.y[1], zorder=3)
    
    # integrate from t0 to tau_alpha
    # sol_tau2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, t2), state8, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_tau2.y[0],  sol_tau2.y[1], zorder=3)
    
    # ax3.legend(['$r_1$ to $r_2$', '$r_4$ to $r_5$', '$r_6$ to $r_7$', 
    #             '$r_8$ to $r_9$'])
        
        
    plot_orbit_2D(ax3, LyapIC_L1, P_L1, 'black', xaxis_symmetry=True)
    plot_orbit_2D(ax3, LyapIC_L2, P_L2, 'black', xaxis_symmetry=True)
        
        
        
        
    #%%
    
    # -------------------------------------------------------------------------
    # plot error history
    # -------------------------------------------------------------------------
    
    # start homoclinic connection process
    fig4, ax4 = plt.subplots()
    ax4.set_title('Constraint Vector History')
    # ax4.set_aspect('equal', adjustable='box')
    ax4.grid(alpha=0.2)
    ax4.set_xlabel('iterations')
    ax4.set_ylabel('error (nd)')  
    
    plt.yscale("log")
    ax4.plot(f1)
    ax4.plot(f2)
    ax4.plot(f3)

    
    ax4.legend(['$r_4-r_3$', '$r_6-r_5$', '$r_7-r_{10}$'])  
        
        
        
        
        
        
        
        


    # #%%
    # # -------------------------------------------------------------------------
    # # plot initial guess
    # # -------------------------------------------------------------------------
    
    # # start homoclinic connection process
    # fig3, ax3 = plt.subplots()
    # ax3.set_title('Initial Guess for Heteroclinic Tau-Alpha')
    # ax3.set_aspect('equal', adjustable='box')
    # ax3.grid(alpha=0.2)
    # ax3.set_xlabel('x (nd)')
    # ax3.set_ylabel('y (nd)')
    
    # plt.scatter(state1[0], state1[1], zorder=5)
    # plt.scatter(state2[0], state2[1], zorder=5)
    # plt.scatter(state3[0], state3[1], zorder=5)
    # plt.scatter(state4[0], state4[1], zorder=5)
    # plt.scatter(state5[0], state5[1], zorder=5)
    # plt.scatter(state6[0], state6[1], zorder=5)
    # plt.scatter(state7[0], state7[1], zorder=5)
    # plt.scatter(state8[0], state8[1], zorder=5)
    # plt.scatter(state9[0], state9[1], zorder=5)
    # plt.scatter(state10[0], state10[1], zorder=5)
    # plt.scatter(state11[0], state11[1], zorder=5)
    # plt.scatter(state12[0], state12[1], zorder=5)
    # plt.scatter(state13[0], state13[1], zorder=5)
    # plt.scatter(state14[0], state14[1], zorder=5)
    
    
    # # plt.legend(['$\bar r_1$', '$\bar r_2$', '$\bar r_3$', '$\bar r_4$', '$\bar r_5$', 
    # #             '$\bar r_6$', '$\bar r_7$', '$\bar r_8$', '$\bar r_9$', '$\bar r_10$', 
    # #             '$\bar r_11$', '$\bar r_12$', '$\bar r_13$', '$\bar r_14$'])
    
    # plt.legend(['$r_1$', '$r_2$', '$r_3$', '$r_4$', '$r_5$', 
    #             '$r_6$', '$r_7$', '$r_8$', '$r_9$', '$r_{10}$', 
    #             '$r_{11}$', '$r_{12}$', '$r_{13}$', '$r_{14}$'], loc=2)
    
    
    # plot_orbit_2D(ax3, LyapIC_L1, P_L1, 'black', xaxis_symmetry=True)
    # plot_orbit_2D(ax3, LyapIC_L2, P_L2, 'black', xaxis_symmetry=True)
    # # ax3.plot(sol_Ts.y[0],  sol_Ts.y[1], c='red')
    
    # #%%
    # # unstable half
    # # integrate from t0 to tau_alpha
    # sol_tau1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tau_1), state1, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_tau1.y[0],  sol_tau1.y[1])
    
    # # integrate from tau_1 to T_u1
    # sol_Tu1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, T_u1), state4, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_Tu1.y[0],  sol_Tu1.y[1])
    
    # # integrate from T_u1 to T_u2
    # sol_Tu2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, T_u2), state6, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_Tu2.y[0],  sol_Tu2.y[1])
    
    # #%%
    # # stable half
    # # integrate from t0 to tau_alpha
    # sol_tau2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tau_2), state12, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_tau2.y[0],  sol_tau2.y[1])
    
    # # integrate from tau_1 to T_u1
    # sol_Ts1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_s1), state8, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_Ts1.y[0],  sol_Ts1.y[1])
    
    # # integrate from T_u1 to T_u2
    # sol_Ts2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_s2), state10, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_Ts2.y[0],  sol_Ts2.y[1])
    
    
    
    
    
    
    
    
    # #%%
    # # get middle state and time for unstable manifold
    # T_u1 = find_middle(sol_Tu.t)
    # state5 = np.array([find_middle(sol_Tu.y[i]) for i in range(6)])
    
    # # integrate from t0 to tau_alpha
    # # sol_tau2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tau_2), LyapIC_L2, 
    # #                 dense_output=True, rtol=tol, atol=tol)
    # # ax3.plot(sol_tau2.y[0],  sol_tau2.y[1], c='black')
    # # ax3.plot(sol_tau2.y[0], -sol_tau2.y[1], c='black')
    
    # # integrate from tau_alpha to tau_M
    # sol_Ts = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, T_s),   stableIC[stidxs], 
    #                 dense_output=True, rtol=tol, atol=tol)
    
    
    # # get middle state and time for   stable manifold
    # T_s1 = find_middle(sol_Ts.t)
    # state9 = np.array([find_middle(sol_Ts.y[i]) for i in range(6)])
    
    # # ax3.legend(['Segment 1', 'Segment 2'])
    
    # # # add the moon
    # # rmoon = system_data['r moon']
    # # moon = plt.Circle((1-mu, 0), rmoon, color='gray', zorder=5)
    # # ax3.add_patch(moon)
    
    # # add the libration points
    # ax3.scatter(xLib[0], yLib[0], marker='*', color='yellow', zorder=5)
    # ax3.scatter(xLib[1], yLib[1], marker='*', color='yellow', zorder=5)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # #%% OLD
    # # -------------------------------------------------------------------------
    # # plot initial guess
    # # -------------------------------------------------------------------------
    
    # # start homoclinic connection process
    # fig3, ax3 = plt.subplots()
    # ax3.set_title('Initial Guess for Heteroclinic Tau-Alpha')
    # ax3.set_aspect('equal', adjustable='box')
    # ax3.grid(alpha=0.2)
    # ax3.set_xlabel('x (nd)')
    # ax3.set_ylabel('y (nd)')
    
    # plot_orbit_2D(ax3, LyapIC_L1, P_L1, 'black', xaxis_symmetry=True)
    # plot_orbit_2D(ax3, LyapIC_L2, P_L2, 'black', xaxis_symmetry=True)
    
    # # integrate from t0 to tau_alpha
    # # sol_tau1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P_L1), LyapIC_L1, 
    # #                 dense_output=True, rtol=tol, atol=tol)
    # # ax3.plot(sol_tau1.y[0],  sol_tau1.y[1], c='black')
    # # ax3.plot(sol_tau1.y[0], -sol_tau1.y[1], c='black')
    
    # # integrate from tau_alpha to tau_M
    # sol_Tu = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, T_u), unstableIC[unidxs], 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_Tu.y[0], sol_Tu.y[1], c='red')
    
    # # get middle state and time for unstable manifold
    # T_u1 = find_middle(sol_Tu.t)
    # state5 = np.array([find_middle(sol_Tu.y[i]) for i in range(6)])
    
    # # integrate from t0 to tau_alpha
    # # sol_tau2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tau_2), LyapIC_L2, 
    # #                 dense_output=True, rtol=tol, atol=tol)
    # # ax3.plot(sol_tau2.y[0],  sol_tau2.y[1], c='black')
    # # ax3.plot(sol_tau2.y[0], -sol_tau2.y[1], c='black')
    
    # # integrate from tau_alpha to tau_M
    # sol_Ts = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, T_s),   stableIC[stidxs], 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_Ts.y[0], sol_Ts.y[1], c='blue')
    
    # # get middle state and time for   stable manifold
    # T_s1 = find_middle(sol_Ts.t)
    # state9 = np.array([find_middle(sol_Ts.y[i]) for i in range(6)])
    
    # # ax3.legend(['Segment 1', 'Segment 2'])
    
    # # add the moon
    # rmoon = system_data['r moon']
    # moon = plt.Circle((1-mu, 0), rmoon, color='gray', zorder=5)
    # ax3.add_patch(moon)
    
    # # add the libration points
    # ax3.scatter(xLib[0], yLib[0], marker='*', color='yellow', zorder=5)
    # ax3.scatter(xLib[1], yLib[1], marker='*', color='yellow', zorder=5)