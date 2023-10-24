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
from tools import normalize, set_axes_equal, get_manifold_ICs_2D
from targeters import xzplane_xfixed_2D
import multiproc_integration_2D as multi


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
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(4)
    phi0 = phi0.reshape(1, 16)
    
    xLib, yLib = libsolver(mu, tol)
    L1 = np.array([xLib[0], yLib[0], 0.0, 0.0])
    L2 = np.array([xLib[1], yLib[1], 0.0, 0.0])
    L3 = np.array([xLib[2], yLib[2], 0.0, 0.0])
    L4 = np.array([xLib[3], yLib[3], 0.0, 0.0])
    L5 = np.array([xLib[4], yLib[4], 0.0, 0.0])
    
    LyapIC = np.array([0.9569, 0.0, 0.0, -0.8839])
    LyapIC = np.array([0.8879, 0.0, 0.0, -0.3357])
    LyapIC = np.array([0.8554, 0.0, 0.0, -0.1379])
    
    orbitIC, t_cross = xzplane_xfixed_2D(LyapIC)
    P = t_cross*2
    
    num_manifolds = 200
    stableIC,   stTpoints = get_manifold_ICs_2D(mu, orbitIC, P, stepoff, num_manifolds, 
                               positive_dir=True, stable=True)
    unstableIC, unTpoints = get_manifold_ICs_2D(mu, orbitIC, P, stepoff, num_manifolds, 
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
    integ_states_to_moon = p.map(multi.stable_to_secondary, stableIC)
    # index state portion of solve_ivp solution
    stable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # index time portion of solve_ivp solution
    stable_times_to_moon = [integ_states_to_moon[i].t[-1] for i in range(len(integ_states_to_moon))]
    # plot and get final state at Lunar crossing
    stable_states_at_Moon = plot_manifolds_2D(ax1, stable_states_to_moon, 'blue')
    
    # integrate stable manifolds from Lunar crossing to xaxis crossing
    integ_states_to_xaxis = p.map(multi.stable_to_xaxis, stable_states_at_Moon)
    # index state portion of solve_ivp solution
    stable_states_to_xaxis = [integ_states_to_xaxis[i].y for i in range(len(integ_states_to_xaxis))]
    # index time portion of solve_ivp solution
    stable_times_to_xaxis = [integ_states_to_xaxis[i].t[-1] for i in range(len(integ_states_to_xaxis))]
    # plot and get final state at x-axis crossing
    stable_states_at_xaxis = plot_manifolds_2D(ax1, stable_states_to_xaxis, 'blue')
    
    #%%
    # integrate stable manifolds to Moon and get final state at Lunar crossing
    integ_states_to_moon = p.map(multi.unstable_to_secondary, unstableIC)
    # index state portion of solve_ivp solution
    unstable_states_to_moon = [integ_states_to_moon[i].y for i in range(len(integ_states_to_moon))]
    # index time portion of solve_ivp solution
    unstable_times_to_moon = [integ_states_to_moon[i].t[-1] for i in range(len(integ_states_to_moon))]
    # plot and get final state at Lunar crossing
    unstable_states_at_Moon = plot_manifolds_2D(ax1, unstable_states_to_moon, 'red')
    
    # integrate stable manifolds from Lunar crossing to xaxis crossing
    integ_states_to_xaxis = p.map(multi.unstable_to_xaxis, unstable_states_at_Moon)
    # index state portion of solve_ivp solution
    unstable_states_to_xaxis = [integ_states_to_xaxis[i].y for i in range(len(integ_states_to_xaxis))]
    # index time portion of solve_ivp solution
    unstable_times_to_xaxis = [integ_states_to_xaxis[i].t[-1] for i in range(len(integ_states_to_xaxis))]
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
    ax2.set_title('Poincare Map')
    # ax2.add_patch(moon)
    poincare_section(ax2,   stable_states_at_xaxis, 0, 2, 'blue')
    poincare_section(ax2, unstable_states_at_xaxis, 0, 2, 'red')
    
    #%%
    
    # -------------------------------------------------------------------------
    # free variable vector
    # -------------------------------------------------------------------------
    
    # get initial guess for manifold that terminates closest to perpendicular
    idxs = list(range(0, len(unstable_states_at_xaxis)))
    # want x-velocity to be 0
    idxs.sort(key = lambda x:np.abs(unstable_states_at_xaxis[x][2]))
    
    # initial guess for tau-alpha will be the Tpoint corresponding to idxs[0]
    tau_alpha = unTpoints[idxs[0]]
    
    # initial guess for tau-M is the total integration time along manifold
    tau_M = unstable_times_to_moon[idxs[0]] + unstable_times_to_xaxis[idxs[0]]
    
    # initial guess for final manifold state at axis crossing
    state5 = unstable_states_at_xaxis[idxs[0]]
    
    # propagate state5 backwards in time for t=tau_M to get guess for state4
    sol4 = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -tau_M), state5, 
                    dense_output=True, rtol=tol, atol=tol)
    
    state4 = np.array([sol4.y[i][-1] for i in range(len(sol4.y))])
    
    # free variable vector
    X = np.concatenate((state4, [tau_alpha], [tau_M]), axis=0)
    
    
    # -------------------------------------------------------------------------
    # plot initial guess
    # -------------------------------------------------------------------------
    
    # # start homoclinic connection process
    # fig3, ax3 = plt.subplots()
    # ax3.set_title('Initial Guess for Tau-Alpha')
    # ax3.set_aspect('equal', adjustable='box')
    # ax3.grid(alpha=0.2)
    # ax3.set_xlabel('x (nd)')
    # ax3.set_ylabel('y (nd)')
    
    # # integrate from t0 to tau_alpha
    # sol_alpha = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tau_alpha), orbitIC, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_alpha.y[0], sol_alpha.y[1])
    
    # # integrate from tau_alpha to tau_M
    # sol_M = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, -tau_M), state5, 
    #                 dense_output=True, rtol=tol, atol=tol)
    # ax3.plot(sol_M.y[0], sol_M.y[1])
    
    # ax3.legend(['Segment 1', 'Segment 2'])

    
    
    # def homoclinic_update(X, orbitIC, P):
        
    error = 1
    Xnew = X
    iterations = 1
    error_history = []
    
    while error > tol:
        X = Xnew
        
        # -------------------------------------------------------------------------
        # constraint vector
        # -------------------------------------------------------------------------
        
        # initial guesses from free variable vector
        state4 = X[0:4]
        tau_alpha, tau_M = X[4::]
        
        # get manifold states at current tau-alpha guess
        guess_sol = get_manifold_ICs_2D(mu, orbitIC, P, stepoff, 3, 
                                    positive_dir=True, stable=False, 
                                    tau_alpha=tau_alpha,
                                    return_eigvecs=True, return_STMs=True, 
                                    return_fixedpoints=True)
        # unpack manifold results
        unstableIC1, unTpoints1, stvecs, unvecs, STMs, fixedpoints = guess_sol
        
        # stable and unstable eigenvectors of monodromy matrix
        stvec = stvecs[0]
        unvec = unvecs[0]
        
        # fixed point that we step off of to get manifold
        state2 = fixedpoints[1]
        
        # STM on PO at tau-alpha
        STM_alpha = STMs[1]
        
        # state after stepping off PO onto manifold
        state3 = unstableIC1[1]
        
        # add STM to state
        state4 = np.concatenate([state4, phi0[0]])
        
        # integrate from tau_alpha to tau_M
        sol_M = solve_ivp(lambda t, y: EOM_STM_2D(t, y, mu), (t0, tau_M), state4, 
                        dense_output=True, rtol=tol, atol=tol)
        
        # state at x-axis crossingon far side of moon
        state5 = np.array([sol_M.y[i][-1] for i in range(len(sol_M.y))])
        
        # 4x4 identity matrix
        I4 = np.identity(4)
        
        # constraint vector
        const1 = state4[0:4]-state3
        F = np.concatenate((const1, [state5[1]], [state5[2]]), axis=0)
    
        # -------------------------------------------------------------------------
        # jacobian - partial derivative of F wrt X
        # -------------------------------------------------------------------------
        
        # STM at x-axis crossing after integrating for tau-M
        STM_M = state5[4::].reshape((4, 4))
        
        # 2nd row of STM relates changes in final  y to changes in initial state
        phi_M2 = STM_M[1]
        # 4th row of STM relates changes in final vx to changes in initial state
        phi_M4 = STM_M[2]
        
        # derivative of first contraints wrt tau-alpha
        # time derivative of the fixed point that yields the desired manifold
        dstate_STM2 = EOM_STM_2D(0, np.concatenate([state2, STM_alpha.reshape(1, 16)[0]]), mu)
        # state portion
        dstate2 = dstate_STM2[0:4]
        # STM portion
        dSTM_alpha = dstate_STM2[4::].reshape(4,4)
        
        # norm of eigenvector at tau-alpha
        norm = np.linalg.norm(STM_alpha@unvec)
        # splitting math into smaller terms to simplify
        term2 = STM_alpha@unvec*(unvec.T@dSTM_alpha.T@ STM_alpha@unvec + 
                                 unvec.T@ STM_alpha.T@dSTM_alpha@unvec)
        # full derivative
        drdtau = -dstate2 - stepoff*(dSTM_alpha@unvec/norm - term2/(2*norm**3))
        
        # derivative of final state at x-axis crossing
        dstate5 = EOM_2D(0, state5, mu)
        
        # combine to form jacobian
        DF_top = np.concatenate((    I4,      drdtau.reshape(1,4).T), axis=1)
        DF_top = np.concatenate((DF_top, np.array([[0, 0, 0, 0]]).T), axis=1)
        
        DF_r5 = np.concatenate((phi_M2, [0], [dstate5[1]]))
        DF_r6 = np.concatenate((phi_M4, [0], [dstate5[2]]))
        
        DF = np.vstack((DF_top, DF_r5, DF_r6))
        
        update = np.linalg.pinv(DF)@F
        
        
        # update the free variable vector
        Xnew = X - update
        iterations += 1
        error = np.linalg.norm(F)
        error_history.append(error)
        print(F)
        if iterations > 50:
            break
        
    
    
    # F1, X1 = homoclinic_update( X, orbitIC, P)
    # F2, X2 = homoclinic_update(X1, orbitIC, P)
    # F3, X3 = homoclinic_update(X2, orbitIC, P)
    # F4, X4 = homoclinic_update(X3, orbitIC, P)
    # F5, X5 = homoclinic_update(X4, orbitIC, P)
    # F6, X6 = homoclinic_update(X5, orbitIC, P)
    # F7, X7 = homoclinic_update(X6, orbitIC, P)
    # F8, X8 = homoclinic_update(X7, orbitIC, P)
    # F9, X9 = homoclinic_update(X8, orbitIC, P)
    
    
    
    
    # #%%
    # # -------------------------------------------------------------------------
    # # plot guess
    # # -------------------------------------------------------------------------
    
    # # start homoclinic connection process
    # fig3, ax3 = plt.subplots()
    # ax3.set_title('Initial Guess for Tau-Alpha')
    # ax3.set_aspect('equal', adjustable='box')
    # ax3.grid(alpha=0.2)
    # ax3.set_xlabel('x (nd)')
    # ax3.set_ylabel('y (nd)')
    
    # def plot_guesses(ax, X, orbitIC):
    #     state = X[0:4]
    #     tau_alpha, tau_M = X[4::]
        
    #     # integrate from t0 to tau_alpha
    #     sol_alpha = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tau_alpha), orbitIC, 
    #                     dense_output=True, rtol=tol, atol=tol)
    #     ax.plot(sol_alpha.y[0], sol_alpha.y[1])
        
    #     # integrate from tau_alpha to tau_M
    #     sol_M = solve_ivp(lambda t, y: EOM_2D(t, y, mu), (t0, tau_M), state, 
    #                     dense_output=True, rtol=tol, atol=tol)
    #     ax.plot(sol_M.y[0], sol_M.y[1])
        
    #     # ax3.legend(['Segment 1', 'Segment 2'])
    # plot_guesses(ax3,  X, orbitIC)
    # plot_guesses(ax3, X1, orbitIC)
    # plot_guesses(ax3, X2, orbitIC)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # #%%
    # # constraint vector should be a function of free variable vector
    # def constraint_homoclinic(orbitIC, P, X):
        
    #     # initial guesses from free variable vector
    #     state4, tau_alpha, tau_M = X
        
    #     # get manifold states at current tau-alpha guess
    #     unstableIC, unTpoints, stvecs, unvecs, STMs = get_manifold_ICs(mu, corr_state, P, stepoff, 3, 
    #                                 positive_dir=True, stable=False, tau_alpha=tau_alpha,
    #                                 return_eigvecs=True, return_STMs=True)
        
    #     # state after stepping off PO onto manifold
    #     state3 = unstableIC[1]
        
    #     # integrate from tau_alpha to tau_M
    #     sol_M = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tau_M), state4, 
    #                     dense_output=True, rtol=tol, atol=tol)
        
    #     # state at x-axis crossingon far side of moon
    #     state5 = [sol_M.y[i][-1] for i in range(6)]
        
    #     # constraint vector
    #     F = np.array([state4-state3, state5[1], state5[3]], dtype=object)
        
    #     return F
        
    # # jacobian is also a function of free variable vector
    # # def jacobian_homoclinic(orbitIC, P, X):
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
        
    
    
    
    
    
    
    















