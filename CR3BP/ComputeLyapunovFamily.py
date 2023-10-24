# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:09:46 2023
@author: sam

get lyapunov orbit families
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import scipy

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

from CR3BP_equations import libsolver, jacobi_constant, EOM, EOM_STM, pseudo_potential
from tools import normalize, sort_eigenvalues
from targeters import xzplane_xfixed

def plot_orbit_2D(ax, orbitIC, P, color, 
                  xaxis_symmetry=False, 
                  return_final_state=False,
                  return_monodromy=False):
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    if return_monodromy==True:
        sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, P), orbitIC, 
                        dense_output=True, rtol=tol, atol=tol)
    else:
        sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P), orbitIC, 
                        dense_output=True, rtol=tol, atol=tol)
    
    ax.plot(sol.y[0],  sol.y[1], c=color)
    
    # plot orbit symmetrix about x-axis
    if xaxis_symmetry==True:
        ax.plot(sol.y[0], -sol.y[1], c=color)
    # return final integrated state
    if return_final_state==True:
        return np.array([sol.y[i][-1] for i in range(len(sol.y))])

def linear_approx(Lpoint):
    
    # perturbation from initial state
    xi0  = 0.01

    # pseudo potential and its partials
    U, Ux, Uy, Uz, Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz = pseudo_potential(mu, Lpoint[0:3])
    
    b1 = 2 - (Uxx + Uyy)/2
    b2 = np.sqrt(-Uxx*Uyy)

    # linear approximation
    s  = (b1 + (b1**2 + b2**2)**(1/2))**(1/2)
    b3 = (s**2 + Uxx)/(2*s)

    # perturbation to the state components based on linear approximation
    x0  =  Lpoint[0] + xi0
    vy0 = -b3*(xi0)*s
    
    return np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
        

#%%
if __name__ == '__main__':
    
    # ------------------ general stuff
    
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
    xpoints, ypoints = libsolver(mu, tol) 
    
    # ------------------ things to edit based on desired family/libration point
    
    # matricies for transforming the half period STM to the monodromy matrix
    # based on symmetry of desired family
    A = np.array([[ 1,  0,  0,  0,  0,  0],
                  [ 0, -1,  0,  0,  0,  0],
                  [ 0,  0,  1,  0,  0,  0],
                  [ 0,  0,  0, -1,  0,  0],
                  [ 0,  0,  0,  0,  1,  0],
                  [ 0,  0,  0,  0,  0, -1]])
    
    J = np.array([[ 0,  0,  0,  1,  0,  0],
                  [ 0,  0,  0,  0,  1,  0],
                  [ 0,  0,  0,  0,  0,  1],
                  [-1,  0,  0,  0,  2,  0],
                  [ 0, -1,  0, -2,  0,  0],
                  [ 0,  0, -1,  0,  0,  0]])
    
    # determine which libration points and primaries to include in plot
    pltL1=True
    pltL2=True
    pltL3=False
    pltL4=False
    pltL5=False
    pltPri=False
    pltSec=True
    
    # path to folder to save data
    filepath = r'C:\Users\sam\Desktop\Research\CR3BP\OrbitFamilies\EarthMoon\Lyapunovs'
    resave = False
    
    # replot family (involves a lot of integration)
    plot_fam = False
    
    L1 = True
    L2 = False
    L3 = False
    
    if L1==True:
        
        # libration point of interest (index 0-4)
        Lpoint_indx = 0
        
        # determine step size
        # positive - step to the right, negative - step to left
        dx = 0.0001
        # terminate targeting integration when going from positive to negative boolean
        pos2neg = False
        
        # total number of steps to take
        num_steps = 1360
        
        # name of final data file for excel export
        filename = r'\L1Lyapunovs.xlsx'
        
        # libration point of interest
        Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
        
        # initial conditions for a Lyapunov orbit very close to L1
        LyapIC_L1 = linear_approx(Lpoint)
        LyapIC_L1, P_L1, STM_L1 = xzplane_xfixed(LyapIC_L1, pos2neg=pos2neg, return_STM=True)
        
        # transform half period STM to get monodromy matrix
        Mon_L1 = A@J@STM_L1.T@np.linalg.inv(J)@A@STM_L1
        
        # redefine based on chosen starting point
        LyapIC, P, Mon = LyapIC_L1, P_L1, Mon_L1
    
    if L2==True:
        # determine which libration points and primaries to include in plot
        pltL1=True
        pltL2=True
        pltL3=False
        pltL4=False
        pltL5=False
        pltPri=False
        pltSec=True
        
        # libration point of interest (index 0-4)
        Lpoint_indx = 1
        
        # determine step size
        # positive - step to the right, negative - step to left
        dx = -0.0001
        # terminate targeting integration when going from positive to negative boolean
        pos2neg = True
        
        # total number of steps to take
        num_steps = 1500
        
        # name of final data file for excel export
        filename = r'\L2Lyapunovs.xlsx'
        
        # libration point of interest
        Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
        
        # initial conditions for a Lyapunov orbit very close to L2
        LyapIC_L2 = linear_approx(Lpoint)
        LyapIC_L2, P_L2 = xzplane_xfixed(LyapIC_L2)
        JC_L2 = jacobi_constant(mu, LyapIC_L2)
        
        # this gets initial conditions to the right of L2, but we want to be on
        # the left between the libration point and the moon
        # propagate to the next axis crossing and get conditions on other side
        
        # 2D plot formatting
        fig, ax = plt.subplots()
        LyapFC_L2 = plot_orbit_2D(ax, LyapIC_L2, P_L2, 'black', 
                          xaxis_symmetry=False, return_final_state=True)
        
        # recorrect based on chosen starting point
        LyapIC, P, STM_L2 = xzplane_xfixed(LyapFC_L2, pos2neg=pos2neg, return_STM=True)
        
        # transform half period STM to get monodromy matrix
        Mon = A@J@STM_L2.T@np.linalg.inv(J)@A@STM_L2
        
    if L3==True:
        # determine which libration points and primaries to include in plot
        pltL1=True
        pltL2=True
        pltL3=True
        pltL4=True
        pltL5=True
        pltPri=True
        pltSec=True
        
        # libration point of interest (index 0-4)
        Lpoint_indx = 2
        
        # determine step size
        # positive - step to the right, negative - step to left
        dx = 0.001
        # terminate targeting integration when going from positive to negative boolean
        pos2neg = False
        
        # total number of steps to take
        num_steps = 960
        
        # name of final data file for excel export
        filename = r'\L3Lyapunovs.xlsx'
        
        # libration point of interest
        Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
        
        # initial conditions for a Lyapunov orbit very close to L1
        LyapIC_L3 = linear_approx(Lpoint)
        LyapIC_L3, P_L3, STM_L3 = xzplane_xfixed(LyapIC_L3, return_STM=True)
        
        # transform half period STM to get monodromy matrix
        Mon_L3 = A@J@STM_L3.T@np.linalg.inv(J)@A@STM_L3
        
        # recorrect based on chosen starting point
        LyapIC, P, Mon = LyapIC_L3, P_L3, Mon_L3
    
    
    
    
    #%% SHOULD BE GENERALIZED FROM HERE DOWN FOR ANY LYAPUNOV FAMILY
    
    # ------------------ natural parameter continuation to get initial conditions
    
    # jacobi constant of initial orbit from linear approximation
    JC = jacobi_constant(mu, LyapIC)
    
    # same previous step to get slope for initial guess at new step
    prevIC = LyapIC
    prevP  = P
    
    # get state and period for next orbit in family
    currIC = np.array([LyapIC[0] + dx, 0, 0, 0, LyapIC[4], 0])
    currIC, currP, currSTM = xzplane_xfixed(currIC, pos2neg=pos2neg, return_STM=True)
    currJC = jacobi_constant(mu, currIC)
    
    # transform half period STM to get monodromy matrix
    currMon = A@J@currSTM.T@np.linalg.inv(J)@A@currSTM
    
    # keep record of states periods and JC for each orbit
    states = [LyapIC, currIC]
    times  = [P, currP]
    JCs    = [JC, currJC]
    
    # get eigenvalues and eigenvectors
    vals1, vecs1 = scipy.linalg.eig(Mon)
    vals2, vecs2 = scipy.linalg.eig(currMon)
    
    # store history of monodromy matrix, eigvals, stability index, and time constant
    MonMs, unsorted_vals, unsorted_vecs = [Mon, currMon], [vals1, vals2], [vecs1, vecs2]
    
    # step through family and get corrected IC for each orbit
    for i in range(num_steps):
        # print something to show progress
        if i%10 == 0:
            print('currently on step {}/{} of L{} family'.format(i, num_steps, Lpoint_indx+1))
        
        # x and vy for previous 2 orbits to get initial guess for new orbit
        x1  = prevIC[0]
        x2  = currIC[0]
        vy1 = prevIC[4]
        vy2 = currIC[4]
        
        m = (vy2 - vy1)/(x2 - x1)
        b =  vy1 - m*x1
        
        xnew  = x2 + dx
        vynew = m*xnew + b
        
        state1 = np.array([xnew, 0, 0, 0, vynew, 0])
        LyapIC, P, STM = xzplane_xfixed(state1, pos2neg=pos2neg, return_STM=True)
        JC = jacobi_constant(mu, LyapIC)
        
        # transform half period STM to get monodromy matrix
        Mon = A@J@STM.T@np.linalg.inv(J)@A@STM
        
        # get eigenvalues and eigenvectors
        vals, vecs = scipy.linalg.eig(Mon)
                             
        # add new values to history list
        states.append(LyapIC)
        times.append(P)
        JCs.append(JC)
        MonMs.append(Mon)
        unsorted_vals.append(vals)
        unsorted_vecs.append(vecs)
        
        prevIC = currIC
        currIC = LyapIC
    

    
    
    #%%
    if plot_fam==True:
        '''
        plot the Lyapunov family
        '''
        plt.style.use(['default'])
        title = 'Earth-Moon L$_{}$ Lyapunov Family'.format(Lpoint_indx+1)
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        
        # plot formatting
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.grid(alpha=0.2)
        
        # x and y axis labels
        ax.set_xlabel('x (nd)')
        ax.set_ylabel('y (nd)')
        
        # plot the libration point locations
        if pltL1==True:
            ax.scatter(xpoints[0], ypoints[0], zorder=3, color = 'goldenrod', marker = '*')
        if pltL2==True:
            ax.scatter(xpoints[1], ypoints[1], zorder=3, color = 'goldenrod', marker = '*')
        if pltL3==True:
            ax.scatter(xpoints[2], ypoints[2], zorder=3, color = 'goldenrod', marker = '*')
        if pltL4==True:
            ax.scatter(xpoints[3], ypoints[3], zorder=3, color = 'goldenrod', marker = '*')
        if pltL5==True:
            ax.scatter(xpoints[4], ypoints[4], zorder=3, color = 'goldenrod', marker = '*')
        
        # plot the larger primary location
        if pltPri==True:
            ax.scatter( -mu, 0.0, zorder=2, color = 'green')
        
        # plot the smaller primary location
        if pltSec==True:
            ax.scatter(1-mu, 0.0, zorder=2, color = 'gray')
    
        # Jacobi Constant color mapping
        cmap = plt.cm.cool
        norm = plt.Normalize(vmin=np.min(JCs), vmax=np.max(JCs))
        cbar_ax = fig.add_axes([0.65, 0.11, 0.02, 0.76])
        cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, label='Jacobi Constant', 
                                        norm=norm, orientation='vertical')
        
        # store history of monodromy matrix, eigvals, stability index, and time constant
        STMs, unsorted_vals, unsorted_vecs = [], [], []
        
        # iterate through each orbit in list
        for i in range(len(states)):
            # only plot every 20 orbits so theyre spaced out to visualize
            if i%20 == 0:
            
                # extract current states, times, and JC
                state = np.concatenate([states[i], phi0[0]])
                tf = times[i]*2
                JC = jacobi_constant(mu, states[i])
                
                # integrate the orbit for the full period
                sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state,
                                dense_output = True, rtol = tol, atol = tol)
                
                # plot orbit color mapped to JC
                ax.plot(sol.y[0], sol.y[1], color=cmap(norm(JC)), linewidth=0.8)
     
    #%%
    
    # sort the eigenvalues and eigenvectors
    sorted_vals, sorted_vecs = sort_eigenvalues(unsorted_vals, unsorted_vecs)
    
    #%%
    # also store alpha and beta terms for BSD
    stab_idxs, time_consts = [], []
    alphas, betas = [], []
    Lyap_exp  = []
    Lyap_exp2 = []
    comp_vals = []
    
    # now that we have all the monodromy matricies we can iterate through them
    for i in range(len(MonMs)):
        
        # period of current orbit
        t = times[i]*2 
        
        # monodromy matrix of current orbit
        monod = MonMs[i]
        
        # compute stability indicies (zimovan 2021 pg 68)
        abs_val = [abs(v) for  v in sorted_vals[i]]
        stab  = np.array([(1/2)*(j + 1/j) for j in abs_val])
        
        # compute time constant from max eigenvalue (zimovan 2021 pg 71)
        max_val = max(sorted_vals[i])
        nat_val = np.log(max_val)
        t_cons  = 1/(abs(np.real(nat_val))*t)
        
        # Lyapunov exponents lambda = e^(alpha*t)
        # e^x=y ---> ln(y)=x
        exps = [np.log(v)/t for  v in sorted_vals[i]]
        
        # alpha and beta for broucke stability diagram
        alpha = 2 - np.trace(monod)
        beta  = (1/2)*(alpha**2 + 2 - np.trace(monod@monod))
        
        # calculate nontrivial eigenvalues based on JPL paper
        p = (alpha + (alpha**2 - 4*beta + 8)**(0.5))/2
        q = (alpha - (alpha**2 - 4*beta + 8)**(0.5))/2
        
        # assign as complex to avoid issues with negative square roots
        p = complex(p)
        q = complex(q)
        
        lam1 = (-p + (p**2 - 4)**(0.5))/2
        lam2 = (-p - (p**2 - 4)**(0.5))/2
        
        lam3 = (-q + (q**2 - 4)**(0.5))/2
        lam4 = (-q - (q**2 - 4)**(0.5))/2
        
        # trivial eigenvalues
        lam5, lam6 = complex(1.0), complex(1.0)
        
        vals = [lam1, lam2, lam3, lam4, lam5, lam6]
        
        # Lyapunov exponents lambda = e^(alpha*t)
        # e^x=y ---> ln(y)=x
        exps2 = [np.log(v)/t for  v in vals]
        
        # add all values to their respective lists
        stab_idxs.append(stab)
        time_consts.append(t_cons)
        alphas.append(alpha)
        betas.append(beta)
        Lyap_exp.append(exps)
        Lyap_exp2.append(exps2)
        comp_vals.append(vals)
    
    
    #%%
    # pass corrected initial conditions, period, JC, and other values to a DF
    ICData = pd.DataFrame({'initial condition': states, 'P': times, 'JC': JCs, 
                            'time constant': time_consts,
                            'alpha': alphas, 'beta': betas})
    
    # eigenvales and vectors
    valData = pd.DataFrame(sorted_vals, columns=['val 1', 'val 2', 'val 3',
                                                 'val 4', 'val 5', 'val 6'])
    
    valData2 = pd.DataFrame(comp_vals, columns=['analytic eigenvalue 1', 'analytic eigenvalue 2', 'analytic eigenvalue 3',
                                                'analytic eigenvalue 4', 'analytic eigenvalue 5', 'analytic eigenvalue 6'])
    
    vecData = pd.DataFrame(sorted_vecs, columns=['vec1', 'vec2', 'vec3',
                                                 'vec4', 'vec5', 'vec6'])
    
    # stability indicies 
    stbData = pd.DataFrame(stab_idxs, columns=['stb1', 'stb2', 'stb3',
                                               'stb4', 'stb5', 'stb6'])
    
    # Lyapunov exponents
    expData = pd.DataFrame(Lyap_exp, columns=['Lyap exp 1', 'Lyap exp 2', 'Lyap exp 3',
                                              'Lyap exp 4', 'Lyap exp 5', 'Lyap exp 6'])
    
    # Lyapunov exponents
    expData2 = pd.DataFrame(Lyap_exp2, columns=['analytic Lyap exp 1', 'analytic Lyap exp 2', 'analytic Lyap exp 3',
                                                'analytic Lyap exp 4', 'analytic Lyap exp 5', 'analytic Lyap exp 6'])
    
    # add monodromy matrix to eigenvector data
    vecData['monodromy matrix'] = pd.Series(MonMs)
    
    FamilyData = ICData.join([stbData, expData, valData, vecData])
    
    #%%
    # send to excel file
    if resave==True:
        FamilyData.to_excel(filepath + filename)
    
    #%%
    
    # plot time constant
    fig1, ax1 = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8, 9),)
    fig1.set_tight_layout(True)
    
    # plot title
    ax1[0].set_title('Earth-Moon L$_{}$ Lyapunov Family'.format(Lpoint_indx+1))
    
    # half period
    ax1[0].plot(JCs, times, c='magenta')
    
    # time constant
    ax1[1].plot(JCs, time_consts, c='magenta')
    # ax1[1].set_ylim(0, 0.7)
    
    # stabillity index
    ax1[2].plot(JCs, FamilyData['stb1'], c='magenta')
    ax1[2].plot(JCs, FamilyData['stb2'], c='magenta')
    ax1[2].plot(JCs, FamilyData['stb3'], c='magenta')
    ax1[2].plot(JCs, FamilyData['stb4'], c='magenta')
    ax1[2].plot(JCs, FamilyData['stb5'], c='magenta')
    ax1[2].plot(JCs, FamilyData['stb6'], c='magenta')
    
    # Lyapunov exponents
    ax1[3].plot(JCs, expData2['analytic Lyap exp 1'], c='magenta')
    ax1[3].plot(JCs, expData2['analytic Lyap exp 2'], c='magenta')
    ax1[3].plot(JCs, expData2['analytic Lyap exp 3'], c='magenta')
    ax1[3].plot(JCs, expData2['analytic Lyap exp 4'], c='magenta')
    ax1[3].plot(JCs, expData2['analytic Lyap exp 5'], c='magenta')
    ax1[3].plot(JCs, expData2['analytic Lyap exp 6'], c='magenta')
    
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
    
    

    
    

    
    
    
    
    
    
    
    
    
    
    
    