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

from CR3BP_equations import EOM, EOM_STM, libsolver, jacobi_constant
from tools import normalize, set_axes_equal, get_manifold_ICs
from targeters import xzplane_xfixed, xzplane_JCfixed
import multiproc_integration as multi


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
    
def plot_orbit_2D(ax, orbitIC, P, color, xaxis_symmetry=False, ls=None):
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    if ls == None:
        ls = 'solid'
        
    ax.plot(sol.y[0],  sol.y[1], c=color, linestyle=ls)
    if xaxis_symmetry==True:
        ax.plot(sol.y[0], -sol.y[1], c=color, linestyle=ls)
    
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
    # plot_orbit_2D(ax1, LyapIC_L2, P_L2*2, 'blue', xaxis_symmetry=False)

    
    #%%
    # target an L2 orbit with exactly the same JC as the L1 orbit
    LyapIC_L2, P_L2 = xzplane_JCfixed(LyapIC_L2, P_L2, JC_L1)
    JC_L2 = jacobi_constant(mu, LyapIC_L2)
    
    # plot orbit with corrected JC
    plot_orbit_2D(ax1, LyapIC_L2, P_L2, 'cyan', xaxis_symmetry=True)
    
    ax1.legend(['$L_1$ Lyapunov JC = 3.144098',
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
    poincare_section(ax2,   stable_states_at_Moon, 1, 3, 'blue')
    poincare_section(ax2, unstable_states_at_Moon, 1, 3, 'red')
    
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
    
    # initial guess for tau-M is the total integration time along manifold
    T_u = unstable_times_to_moon[unidxs] 
    T_s =   stable_times_to_moon[stidxs] 
    
    
    #%%
    # initial conditions for states along the first periodic orbit
    # state1 is the L1 orbit initiall state
    state1 = LyapIC_L1
    
    # state2 is the fixed point propagated from state1 for t=tau_1 before step off
    # get manifold states at current tau-alpha guess
    un_guess = get_manifold_ICs(mu, state1, P_L1*2, stepoff, 3, 
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
    sol_Tu = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, T_u), state3, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # get middle state and time for first leg of unstable manifold
    T_u1 = find_middle(sol_Tu.t)
    state5 = np.array([find_middle(sol_Tu.y[i]) for i in range(6)])
    
    # state7 is the state at the end of the unstable manifold porpagation
    state7 = np.array([sol_Tu.y[i][-1] for i in range(6)])
    # time along second leg is total time - time along first leg
    T_u2 = sol_Tu.t[-1] - T_u1
    
    # propagate backwards from state5 to get state4
    sol4 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_u1), state5, 
                    dense_output=True, rtol=tol, atol=tol)
    state4 = np.array([sol4.y[i][-1] for i in range(6)])
    
    # propagate backwards from state7 to get state6
    sol7 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_u2), state7, 
                    dense_output=True, rtol=tol, atol=tol)
    state6 = np.array([sol7.y[i][-1] for i in range(6)])

    # initial conditions for states along the second periodic orbit
    # state12 is the L2 orbit initial state
    state12 = LyapIC_L2
    
    # get manifold states at current tau-alpha guess
    st_guess = get_manifold_ICs(mu, state12, P_L2*2, stepoff, 3, 
                                positive_dir=False, stable=True, 
                                tau_alpha=tau_2,
                                return_eigvecs=True, return_STMs=True, 
                                return_fixedpoints=True)
    
    stableIC2, stTpoints2, stvecs2, unvecs2, STMs2, fixedpoints2 = st_guess
    
    # state13 is the fixed point propagated from state12 for t=tau_2 before step off
    state13 = fixedpoints2[1]
    
    # state14 is the initial condition for the stable manifold after stepping off from state13
    state14 = stableIC2[1]
    
    # now need conditions for states along the stable manifold
    # integrate from t0 to T_s
    sol_Ts = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, T_s), state14, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # get middle state and time for first leg of stable manifold
    T_s2 = find_middle(sol_Ts.t)
    state10 = np.array([find_middle(sol_Ts.y[i]) for i in range(6)])
    
    # state8 is the state at the end of the stable manifold porpagation
    state8 = np.array([sol_Ts.y[i][-1] for i in range(6)])
    # time along second leg is total time - time along first leg
    T_s1 = sol_Ts.t[-1] - T_s2
    
    # propagate backwards (in forward time since stable) from state10 to get state11
    sol10 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_s2), state10, 
                    dense_output=True, rtol=tol, atol=tol)
    state11 = np.array([sol10.y[i][-1] for i in range(6)])
    
    # propagate backwards from state8 to get state9
    sol8 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_s1), state8, 
                    dense_output=True, rtol=tol, atol=tol)
    state9 = np.array([sol8.y[i][-1] for i in range(6)])  
    
    
    
    
    # -------------------------------------------------------------------------
    # free variable vector
    # -------------------------------------------------------------------------
    
    X = np.concatenate((state4, state6, state9, state11, [tau_1], [tau_2],
                        [T_u1], [T_u2], [T_s1], [T_s2]), axis=0)
    
    error = 1
    Xnew = X
    iterations = 1
    error_history = []
    f1, f2, f3, f4, f5 = [], [], [], [], []
    
    while error > tol:
        
        # update to free variable vector
        X = Xnew
        
        # unpack free variable vector
        r4  = X[ 0: 6]
        r6  = X[ 6:12]
        r9  = X[12:18]
        r11 = X[18:24]
        t1  = X[24]
        t2  = X[25]
        Tu1 = X[26]
        Tu2 = X[27]
        Ts1 = X[28]
        Ts2 = X[29]
        
        
        # -------------------------------------------------------------------------
        # constraint vector
        # -------------------------------------------------------------------------
        
        # integrate along first orbit segment
        un_guess = get_manifold_ICs(mu, state1, P_L1*2, stepoff, 3, 
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
        sol4 = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, Tu1), 
                         np.concatenate([r4, phi0[0]]), 
                         dense_output=True, rtol=tol, atol=tol)
        sol5 = np.array([sol4.y[i][-1] for i in range(len(sol4.y))])
        r5   = sol5[0:6]
        STM5 = sol5[6::].reshape((6, 6))
        
        
        # integrate along second unstable manifold segment
        sol6 = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, Tu2), 
                         np.concatenate([r6, phi0[0]]), 
                         dense_output=True, rtol=tol, atol=tol)
        sol7 = np.array([sol6.y[i][-1] for i in range(len(sol6.y))])
        r7   = sol7[0:6]
        STM7 = sol7[6::].reshape((6, 6))
        
        
        # integrate along first stable manifold segment
        sol9 = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, Ts1), 
                         np.concatenate([r9, phi0[0]]), 
                         dense_output=True, rtol=tol, atol=tol)
        sol8 = np.array([sol9.y[i][-1] for i in range(len(sol9.y))])
        r8   = sol8[0:6]
        STM8 = sol8[6::].reshape((6, 6))
        
        
        # integrate along second stable manifold segment
        sol11 = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, Ts2),
                          np.concatenate([r11, phi0[0]]), 
                         dense_output=True, rtol=tol, atol=tol)
        sol10 = np.array([sol11.y[i][-1] for i in range(len(sol11.y))])
        r10   = sol10[0:6]
        STM10 = sol10[6::].reshape((6, 6))
        
        
        # integrate along second orbit segment
        st_guess = get_manifold_ICs(mu, state12, P_L2*2, stepoff, 3, 
                                    positive_dir=False, stable=True, 
                                    tau_alpha=t2, full_state=True,
                                    return_eigvecs=True, return_STMs=True, 
                                    return_fixedpoints=True)
        stableIC2, stTpoints2, stvecs2, unvecs2, STMs2, fixedpoints2 = st_guess
        
        # STM and eigevectors for second orbit (at r13)
        STM13 = STMs2[1]
        svec  = stvecs2[0]
        # state13 is the fixed point propagated from state12 for t=tau_2 before step off
        r13 = fixedpoints2[1]
        # state14 is the initial condition for the stable manifold after stepping off from state13
        r14 = stableIC2[1]
        
        
        # constraint vector
        F = np.concatenate((r4-r3, r6-r5, r8-r7, r9-r10, r14-r11), axis=0)
        f1.append(np.linalg.norm( r4 -  r3))
        f2.append(np.linalg.norm( r6 -  r5))
        f3.append(np.linalg.norm( r8 -  r7))
        f4.append(np.linalg.norm( r9 - r10))
        f5.append(np.linalg.norm(r11 - r14))
        
        # -------------------------------------------------------------------------
        # jacobian - partial derivative of F wrt X
        # -------------------------------------------------------------------------
        
        # time derivatives of each state, plus STM for orbit segments
        dr2   = EOM_STM(0, np.concatenate([r2, STM2.reshape(1, 36)[0]]), mu)
        dSTM2 = dr2[6::].reshape((6, 6))
        dr2   = dr2[0:6]
        
        dr5  = EOM(0,  r5, mu)
        dr6  = EOM(0,  r6, mu)
        dr9  = EOM(0,  r9, mu)
        dr11 = EOM(0, r11, mu)
        
        dr13   = EOM_STM(0, np.concatenate([r13, STM13.reshape(1, 36)[0]]), mu)
        dSTM13 = dr13[6::].reshape((6, 6))
        dr13   = dr13[0:6]
        
        # identity matrix
        I6 = np.identity(6)
        # zero matrix
        Z6 = np.zeros(36).reshape(6, 6)
        # zero array
        Z1 = np.zeros(6).reshape(6, 1)
        
        # norm of eigenvector at tau-alpha
        norm2 = np.linalg.norm(STM2@uvec)
        # splitting math into smaller terms to simplify
        term2 = STM2@uvec*(uvec.T@dSTM2.T@ STM2@uvec + 
                           uvec.T@ STM2.T@dSTM2@uvec)
        # full derivative
        dr3dtau = dr2 + stepoff*((dSTM2@uvec)/norm2 - term2/(2*norm2**3))
        
        # norm of eigenvector at tau-alpha
        norm13 = np.linalg.norm(STM13@svec)
        # splitting math into smaller terms to simplify
        term13 = STM13@svec*(svec.T@dSTM13.T@ STM13@svec + 
                             svec.T@ STM13.T@dSTM13@svec)
        # full derivative
        dr14dtau = dr13 + stepoff*((dSTM13@svec)/norm13 - term13/(2*norm13**3))
        
        # form jacobian
        row1 = np.concatenate((I6, Z6, Z6, Z6, 
                              -dr3dtau.reshape(6, 1), Z1, Z1, Z1, Z1, Z1), 
                               axis=1)
        
        row2 = np.concatenate((-STM5, I6, Z6, Z6, 
                               Z1, Z1, -dr5.reshape(6, 1), Z1, Z1, Z1), 
                               axis=1)
        
        row3 = np.concatenate((Z6, -STM7, STM8, Z6, 
                               Z1, Z1, Z1, -dr6.reshape(6, 1), dr9.reshape(6, 1), Z1), 
                               axis=1)
        
        row4 = np.concatenate((Z6, Z6, I6, -STM10, 
                               Z1, Z1, Z1, Z1, Z1, -dr11.reshape(6, 1)), 
                               axis=1)
        
        row5 = np.concatenate((Z6, Z6, Z6, -I6, 
                               Z1, dr14dtau.reshape(6, 1), Z1, Z1, Z1, Z1), 
                               axis=1)
        
        DF = np.vstack((row1, row2, row3, row4, row5))
        
        # update = np.linalg.pinv(DF)@F
        # update = DF.T@(DF@DF.T)**(-1)@F
        update = DF.T@np.linalg.pinv(DF@DF.T)@F
        
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
    state4  = X[ 0: 6]
    state6  = X[ 6:12]
    state8  = X[12:18]
    state10 = X[18:24]
    t1  = X[24]
    t2  = X[25]
    Tu1 = X[26]
    Tu2 = X[27]
    Ts1 = X[28]
    Ts2 = X[29]
    
    # start homoclinic connection process
    fig3, ax3 = plt.subplots()
    ax3.set_title('Final Result for Heteroclinic Tau-Alpha')
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(alpha=0.2)
    ax3.set_xlabel('x (nd)')
    ax3.set_ylabel('y (nd)')
    
    ax3.scatter(r4[0], r4[1], zorder=5)
    ax3.scatter(r6[0], r6[1], zorder=5)
    ax3.scatter(r8[0], r8[1], zorder=5)
    ax3.scatter(r10[0], r10[1], zorder=5)
    ax3.scatter(r11[0], r11[1], zorder=5)
    
    
    # unstable half
    # integrate from t0 to tau_alpha
    sol_tau1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, t1), state1, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_tau1.y[0],  sol_tau1.y[1], zorder=3, c='maroon')
    
    # integrate from tau_1 to T_u1
    sol_Tu1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, Tu1), state4, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_Tu1.y[0],  sol_Tu1.y[1], zorder=3, c='red')
    
    # integrate from T_u1 to T_u2
    sol_Tu2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, Tu2), state6, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_Tu2.y[0],  sol_Tu2.y[1], zorder=3, c='salmon')
    
    
    # stable half
    
    # integrate from tau_1 to T_u1
    sol_Ts1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, Ts1), state8, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_Ts1.y[0],  sol_Ts1.y[1], zorder=3, c='skyblue')
    
    # integrate from T_u1 to T_u2
    sol_Ts2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, Ts2), state10, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_Ts2.y[0],  sol_Ts2.y[1], zorder=4, c='blue')
    
    # integrate from t0 to tau_alpha
    sol_tau2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, t2), state12, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_tau2.y[0],  sol_tau2.y[1], zorder=3, c='midnightblue')
    
    # ax3.legend(['$r_1$ to $r_2$', '$r_4$ to $r_5$', '$r_6$ to $r_7$', 
    #             '$r_8$ to $r_9$', '$r_{10}$ to $r_{11}$', '$r_{12}$ to $r_{13}$'])
        
        
    plot_orbit_2D(ax3, LyapIC_L1, P_L1, 'black', xaxis_symmetry=True, ls='dashed')
    plot_orbit_2D(ax3, LyapIC_L2, P_L2, 'black', xaxis_symmetry=True, ls='dashed')
        
        
        
        
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
    ax4.plot(f4)
    ax4.plot(f5)

    
    ax4.legend(['$r_4-r_3$', '$r_6-r_5$', '$r_8-r_7$', '$r_{10}-r_9$', '$r_{14}-r_{11}$'])  
        
        
        
        
        
        
        
        


    #%%
    # -------------------------------------------------------------------------
    # plot initial guess
    # -------------------------------------------------------------------------
    
    # start homoclinic connection process
    fig3, ax3 = plt.subplots()
    ax3.set_title('Initial Guess for Heteroclinic Tau-Alpha')
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(alpha=0.2)
    ax3.set_xlabel('x (nd)')
    ax3.set_ylabel('y (nd)')
    
    plt.scatter(state1[0], state1[1], zorder=5)
    plt.scatter(state2[0], state2[1], zorder=5)
    plt.scatter(state3[0], state3[1], zorder=5)
    plt.scatter(state4[0], state4[1], zorder=5)
    plt.scatter(state5[0], state5[1], zorder=5)
    plt.scatter(state6[0], state6[1], zorder=5)
    plt.scatter(state7[0], state7[1], zorder=5)
    plt.scatter(state8[0], state8[1], zorder=5)
    plt.scatter(state9[0], state9[1], zorder=5)
    plt.scatter(state10[0], state10[1], zorder=5)
    plt.scatter(state11[0], state11[1], zorder=5)
    plt.scatter(state12[0], state12[1], zorder=5)
    plt.scatter(state13[0], state13[1], zorder=5)
    plt.scatter(state14[0], state14[1], zorder=5)
    
    
    # plt.legend(['$\bar r_1$', '$\bar r_2$', '$\bar r_3$', '$\bar r_4$', '$\bar r_5$', 
    #             '$\bar r_6$', '$\bar r_7$', '$\bar r_8$', '$\bar r_9$', '$\bar r_10$', 
    #             '$\bar r_11$', '$\bar r_12$', '$\bar r_13$', '$\bar r_14$'])
    
    plt.legend(['$r_1$', '$r_2$', '$r_3$', '$r_4$', '$r_5$', 
                '$r_6$', '$r_7$', '$r_8$', '$r_9$', '$r_{10}$', 
                '$r_{11}$', '$r_{12}$', '$r_{13}$', '$r_{14}$'], loc=2)
    
    
    plot_orbit_2D(ax3, LyapIC_L1, P_L1, 'black', xaxis_symmetry=True)
    plot_orbit_2D(ax3, LyapIC_L2, P_L2, 'black', xaxis_symmetry=True)
    # ax3.plot(sol_Ts.y[0],  sol_Ts.y[1], c='red')
    
    #%%
    # unstable half
    # integrate from t0 to tau_alpha
    sol_tau1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tau_1), state1, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_tau1.y[0],  sol_tau1.y[1])
    
    # integrate from tau_1 to T_u1
    sol_Tu1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, T_u1), state4, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_Tu1.y[0],  sol_Tu1.y[1])
    
    # integrate from T_u1 to T_u2
    sol_Tu2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, T_u2), state6, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_Tu2.y[0],  sol_Tu2.y[1])
    
    #%%
    # stable half
    # integrate from t0 to tau_alpha
    sol_tau2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tau_2), state12, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_tau2.y[0],  sol_tau2.y[1])
    
    # integrate from tau_1 to T_u1
    sol_Ts1 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_s1), state8, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_Ts1.y[0],  sol_Ts1.y[1])
    
    # integrate from T_u1 to T_u2
    sol_Ts2 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_s2), state10, 
                    dense_output=True, rtol=tol, atol=tol)
    ax3.plot(sol_Ts2.y[0],  sol_Ts2.y[1])
    
    
    
    
    
    
    
    
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