# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:09:46 2023
@author: sam

get vertical orbit families
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib as mpl
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

from CR3BP_equations import libsolver, jacobi_constant, EOM, EOM_STM, pseudo_potential
from tools import normalize, sort_eigenvalues, set_axes_equal
from targeters import xzplane_xfixed, xzplane_zfixed, xyplane_vyfixed, xyplane_vzfixed
from targeters import xzplane_JCfixed, xzplane_vyfixed, xzplane_vzfixed, xyplane_tfixed

def plot_orbit_3D(ax, orbitIC, P, color, 
                  xyplane_symmetry=False,
                  yzplane_symmetry=False,
                  xzplane_symmetry=False,
                  return_final_state=False):
    
    # padding for axis labels
    pad = 10
    
    # formatting
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)', labelpad=pad)
    ax.set_ylabel('y (nd)', labelpad=pad)
    ax.set_zlabel('z (nd)', labelpad=pad)
    
    # integrate EOM
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # plot for given period
    ax.plot(sol.y[0], sol.y[1], sol.y[2], c=color)
    
    # plot orbit symmetric about planes
    if xyplane_symmetry==True:
        ax.plot(-sol.y[0], -sol.y[1],  sol.y[2], c=color)
    if yzplane_symmetry==True:
        ax.plot(-sol.y[0],  sol.y[1],  sol.y[2], c=color)
    if xzplane_symmetry==True:
        ax.plot( sol.y[0], -sol.y[1],  sol.y[2], c=color)
        
    # return final integrated state
    if return_final_state==True:
        return np.array([sol.y[i][-1] for i in range(6)])
    
def plot_projection_xyplane(ax, orbitIC, P, color, 
                       xsymmetric=False, ysymmetric=False,
                       return_final_state=False):
    # formatting
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    # integrate EOM
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # plot for given period
    ax.plot(sol.y[0], sol.y[1], c=color)
    
    # plot orbit symmetric about planes
    if xsymmetric==True:
        ax.plot( sol.y[0], -sol.y[1], c=color)
    if ysymmetric==True:
        ax.plot(-sol.y[0],  sol.y[1], c=color)
        
    # return final integrated state
    if return_final_state==True:
        return np.array([sol.y[i][-1] for i in range(6)])
    
def plot_projection_xzplane(ax, orbitIC, P, color, 
                       xsymmetric=False, zsymmetric=False,
                       return_final_state=False):
    # formatting
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('z (nd)')
    
    # integrate EOM
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # plot for given period
    ax.plot(sol.y[0], sol.y[2], c=color)
    
    # plot orbit symmetric about planes
    if xsymmetric==True:
        ax.plot( sol.y[0], -sol.y[2], c=color)
    if zsymmetric==True:
        ax.plot(-sol.y[0],  sol.y[2], c=color)
        
    # return final integrated state
    if return_final_state==True:
        return np.array([sol.y[i][-1] for i in range(6)])
    
def plot_projection_yzplane(ax, orbitIC, P, color, 
                       ysymmetric=False, zsymmetric=False,
                       return_final_state=False):
    # formatting
    ax.grid(alpha=0.2)
    ax.set_xlabel('y (nd)')
    ax.set_ylabel('z (nd)')
    
    # integrate EOM
    sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # plot for given period
    ax.plot(sol.y[1], sol.y[2], c=color)
    
    # plot orbit symmetric about planes
    if ysymmetric==True:
        ax.plot( sol.y[1], -sol.y[2], c=color)
    if zsymmetric==True:
        ax.plot(-sol.y[1],  sol.y[2], c=color)
        
    # return final integrated state
    if return_final_state==True:
        return np.array([sol.y[i][-1] for i in range(6)])
    
        

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
    
    # axials are symmetric about x-axis
    G = np.array([[ 1,  0,  0,  0,  0,  0],
                  [ 0, -1,  0,  0,  0,  0],
                  [ 0,  0, -1,  0,  0,  0],
                  [ 0,  0,  0, -1,  0,  0],
                  [ 0,  0,  0,  0,  1,  0],
                  [ 0,  0,  0,  0,  0,  1]])
    
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
    filepath = r'C:\Users\sam\Desktop\Research\CR3BP\OrbitFamilies\EarthMoon\Verticals'
    resave = True
    
    # replot family (involves a lot of integration)
    plot_fam = True
    
    L1 = False
    L2 = True
    L3 = False
    
    if L1==True:
        
        # libration point of interest (index 0-4)
        Lpoint_indx = 0
        
        # determine step size in x and z
        dvz = 0.0010
        dP  = 0.0001
        
        # lower and upper bounds for when to step in vz/vy
        vz_critical  = 1.11
        
        # terminate targeting integration when going from positive to negative boolean
        pos2neg = True
        
        # total number of steps to take
        num_steps = 4000
        
        # name of final data file for excel export
        filename = r'\L1Verticals.xlsx'
        
        # libration point of interest
        Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
        
        # initial conditions for the lyapunov that bifurcates to the halos (grebow 2006 pg 62)
        LyapIC_L1 = np.array([0.8623, 0.0, 0.0, 0.0, 0.0901, 0.4422])
        
        # initial conditions for very small vertical (after continuing inward from Lyap)
        # VertIC_L1 = np.array([0.8377, 0.0, 0.0, 0.0, 0.00152, 0.0717])
        VertIC, P, STM_L1 = xyplane_vzfixed(LyapIC_L1, pos2neg=pos2neg, return_STM=True)
        
        # # 3D plot formatting
        # fig = plt.figure()
        # ax  = fig.add_subplot(111, projection= '3d')
        # VertFC = plot_orbit_3D(ax, VertIC, P, 'black', return_final_state=True)
        # set_axes_equal(ax)
        
        # transform half period STM to get monodromy matrix
        Mon = G@np.linalg.inv(STM_L1)@G@STM_L1
        
    
    if L2==True:
        
        # libration point of interest (index 0-4)
        Lpoint_indx = 1
        
        # determine step size in x and z
        dvz = 0.0010
        dP  = 0.0001
        
        # lower and upper bounds for when to step in vz/vy
        vz_critical  = 1.0245
        
        # terminate targeting integration when going from positive to negative boolean
        pos2neg = True
        
        # total number of steps to take
        num_steps = 4000
        
        # name of final data file for excel export
        filename = r'\L2Verticals.xlsx'
        
        # libration point of interest
        Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
        
        # initial conditions for the lyapunov that bifurcates to the halos (grebow 2006 pg 62)
        LyapIC_L1 = np.array([1.1119, 0.0, 0.0, 0.0, -0.1812, 0.4358])
        
        # initial conditions for very small vertical (after continuing inward from Lyap)
        # VertIC_L1 = np.array([0.8377, 0.0, 0.0, 0.0, 0.00152, 0.0717])
        VertIC, P, STM_L1 = xyplane_vzfixed(LyapIC_L1, pos2neg=pos2neg, return_STM=True)
        
        # 3D plot formatting
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection= '3d')
        VertFC = plot_orbit_3D(ax, VertIC, P, 'black', return_final_state=True)
        set_axes_equal(ax)
        
        # transform half period STM to get monodromy matrix
        Mon = G@np.linalg.inv(STM_L1)@G@STM_L1
        
    if L3==True:
        pltL1=False
        pltL2=False
        pltL3=True
        pltL4=True
        pltL5=True
        pltPri=True
        pltSec=True
        
        # libration point of interest (index 0-4)
        Lpoint_indx = 2
        
        # determine step size in x and z
        dvy =  0.001
        dvz =  0.001
        
        # lower and upper bounds for when to step in z
        vzlow  = 0.64
        vzhigh = 0.31
        
        # terminate targeting integration when going from positive to negative boolean
        pos2neg = True
        
        # total number of steps to take
        num_steps = 945
        
        # name of final data file for excel export
        filename = r'\L3Axials.xlsx'
        
        # libration point of interest
        Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
        
        # initial conditions for the lyapunov that bifurcates to the halos (grebow 2006 pg 40)
        LyapIC_L3 = np.array([-1.8963, 0.0, 0.0, 0.0, 1.6715, 0.0001])
        AxialIC, P, STM_L3 = xyplane_vzfixed(LyapIC_L3, pos2neg=pos2neg, return_STM=True)
        
        # transform half period STM to get monodromy matrix
        Mon = G@J@STM_L3.T@np.linalg.inv(J)@G@STM_L3
        

    # %% SHOULD BE GENERALIZED FROM HERE DOWN FOR ANY VERTICAL FAMILY
    
    # ------------------ natural parameter continuation to get initial conditions
    
    # jacobi constant of initial orbit from linear approximation
    JC = jacobi_constant(mu, VertIC)
    
    # save previous step to get slope for initial guess at new step
    prevIC = VertIC
    prevP  = P
    
    # get state and period for next orbit in family by perturbing in z
    currIC = np.array([VertIC[0], 0, 0, 0, VertIC[4], VertIC[5]+dvz])
    currIC, currP, currSTM = xyplane_vzfixed(currIC, pos2neg=pos2neg, return_STM=True)
    currJC = jacobi_constant(mu, currIC)
    
    # transform half period STM to get monodromy matrix
    currMon = G@np.linalg.inv(currSTM)@G@currSTM
    
    # keep record of states periods and JC for each orbit
    states = [currIC]
    times  = [currP]
    JCs    = [currJC]
    
    vy, vz = [currIC[3]], [currIC[4]]
    
    # get eigenvalues and eigenvectors
    vals1, vecs1 = np.linalg.eig(Mon)
    vals2, vecs2 = np.linalg.eig(currMon)
    
    # store history of monodromy matrix, eigvals, stability index, and time constant
    MonMs, unsorted_vals, unsorted_vecs = [currMon], [vals2], [vecs2]
    
    # initialize both to false
    vzfixed = False
    tfixed = False
    
    # step through family and get corrected IC for each orbit
    for i in range(num_steps):
        # print something to show progress
        if i%20 == 0:
            print('currently on step {}/{} of L{} family'.format(i, num_steps, Lpoint_indx+1))
        
        # state components for current and previous orbits
        x1, y1, z1, vx1, vy1, vz1 = prevIC
        x2, y2, z2, vx2, vy2, vz2 = currIC
        
        if vz2 < 0.01:
            break
        
        # vz-fixed or t-fixed targeting
        z_diff = vz2 - vz_critical
        if z_diff < 0:
            vzfixed = True
            tfixed  = False
                
        else:
            tfixed  = True
            vzfixed = False
            dvz = -0.001
            
        if vzfixed:
            # vy0 slope wrt vz0
            mvz = (vy2-vy1)/(vz2-vz1)
            bvz =  vy1 - mvz*vz1
            
            # step in vz
            vz3 = vz2 + dvz
            # new vy from slope of line
            vy3 = mvz*vz3 + bvz
            
            state1 = np.array([x2, y2, z2, vx2, vy3, vz3])
            VertIC, P, STM = xyplane_vzfixed(state1, pos2neg=pos2neg, return_STM=True)
            
        if tfixed:
            P = P + dP
            VertIC, STM = xyplane_tfixed(state1, P, pos2neg=pos2neg, return_STM=True) 
        
        
        # jacobi constant at current orbit
        JC = jacobi_constant(mu, VertIC)
        
        # transform half period STM to get monodromy matrix
        Mon = G@J@STM.T@np.linalg.inv(J)@G@STM
        
        # get eigenvalues and eigenvectors
        vals, vecs = np.linalg.eig(Mon)
                             
        # add new values to history list
        states.append(VertIC)
        times.append(P)
        JCs.append(JC)
        MonMs.append(Mon)
        unsorted_vals.append(vals)
        unsorted_vecs.append(vecs)
        vy.append(VertIC[3])
        vz.append(VertIC[4])
        
        prevIC = currIC
        currIC = VertIC

    
    #%%
    if plot_fam==True:
        '''
        ------------------------------------------------------------------------------
        plot the Vertical family
        ------------------------------------------------------------------------------
        '''
        plt.style.use(['default'])
        title = 'Earth-Moon L$_{}$ Vertical Family'.format(Lpoint_indx+1)
        
        # 3D plot formatting
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection= '3d')
        
        # plot formatting
        ax.set_title(title)
        
        # x and y axis labels
        ax.set_xlabel('x (nd)')
        ax.set_ylabel('y (nd)')
        ax.set_zlabel('z (nd)')
        
        # plot the libration point locations
        if pltL1==True:
            ax.scatter(xpoints[0], ypoints[0], 0, zorder=3, color = 'goldenrod', marker = '*')
        if pltL2==True:
            ax.scatter(xpoints[1], ypoints[1], 0, zorder=3, color = 'goldenrod', marker = '*')
        if pltL3==True:
            ax.scatter(xpoints[2], ypoints[2], 0, zorder=3, color = 'goldenrod', marker = '*')
        if pltL4==True:
            ax.scatter(xpoints[3], ypoints[3], 0, zorder=3, color = 'goldenrod', marker = '*')
        if pltL5==True:
            ax.scatter(xpoints[4], ypoints[4], 0, zorder=3, color = 'goldenrod', marker = '*')
        
        # plot the larger primary location
        if pltPri==True:
            # 3D earth
            r = system_data['r earth']
            # sphere math
            u = np.linspace(0, 2*np.pi, 100)
            v = np.linspace(0,   np.pi, 100)
            x = r*np.outer(np.cos(u), np.sin(v)) + (-mu)
            y = r*np.outer(np.sin(u), np.sin(v))
            z = r*np.outer(np.ones(np.size(u)), np.cos(v))
            # plot the surface
            ax.plot_surface(x, y, z, color='mediumseagreen', alpha=0.5)
        
        # plot the smaller primary location
        if pltSec==True:
            # 3D moon
            r = system_data['r moon']
            # sphere math
            u = np.linspace(0, 2*np.pi, 100)
            v = np.linspace(0,   np.pi, 100)
            x = r*np.outer(np.cos(u), np.sin(v)) + (1-mu)
            y = r*np.outer(np.sin(u), np.sin(v))
            z = r*np.outer(np.ones(np.size(u)), np.cos(v))
            # plot the surface
            ax.plot_surface(x, y, z, color='gray', alpha=0.5)
    
        # Jacobi Constant color mapping
        cmap = plt.cm.cool
        norm = plt.Normalize(vmin=np.min(JCs), vmax=np.max(JCs))
        cbar_ax = fig.add_axes([0.8, 0.11, 0.02, 0.76])
        cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, label='Jacobi Constant', 
                                        norm=norm, orientation='vertical')
        
        # iterate through each orbit in list
        for i in range(len(states)):
            # only integrate and plot every 20 orbits for visualization
            if i%20 == 0:
                print('plotting {}/{} of L{} family'.format(i, num_steps, Lpoint_indx+1))
            
                # extract current states, times, and JC
                state = states[i]
                tf = times[i]
                JC = jacobi_constant(mu, states[i])
                
                # integrate the orbit for the full period
                sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tf), state,
                                dense_output = True, rtol = tol, atol = tol)
                
                # plot orbit color mapped to JC
                ax.plot(sol.y[0], sol.y[1], sol.y[2], color=cmap(norm(JC)), linewidth=0.8)
        
        set_axes_equal(ax)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(alpha=0.2)
    
    #%%
    
    # sort the eigenvalues and eigenvectors
    sorted_vals, sorted_vecs = sort_eigenvalues(unsorted_vals, unsorted_vecs)
    
    #%%
    # also store alpha and beta terms for BSD
    stab_idxs, time_consts = [], []
    alphas, betas = [], []
    Lyap_exp   = []
    Lyap_exp2  = []
    comp_vals  = []
    stab_idxs2 = []
    
    # now that we have all the monodromy matricies we can iterate through them
    for i in range(len(MonMs)):
        
        # period of current orbit
        t = times[i]*2 
        
        # monodromy matrix of current orbit
        monod = MonMs[i]
        
        # compute stability indicies (zimovan 2021 pg 68)
        abs_val = [abs(v) for v in sorted_vals[i]]
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
        
        # compute stability indicies from analytic eigenvalues
        abs_val = [abs(v) for v in vals]
        stab2 = np.array([(1/2)*(j + 1/j) for j in abs_val])
        
        # add all values to their respective lists
        stab_idxs.append(stab)
        time_consts.append(t_cons)
        alphas.append(alpha)
        betas.append(beta)
        Lyap_exp.append(exps)
        Lyap_exp2.append(exps2)
        comp_vals.append(vals)
        stab_idxs2.append(stab2)
    
    
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
    # stability indicies 
    stbData2 = pd.DataFrame(stab_idxs2, columns=['stb1', 'stb2', 'stb3',
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
    ax1[0].set_title('Earth-Moon L$_{}$ Vertical Family'.format(Lpoint_indx+1))
    
    # half period
    # ax1[0].plot(times, c='magenta')
    ax1[0].plot(JCs, times, c='magenta')
    
    # time constant
    ax1[1].plot(JCs, time_consts, c='magenta')
    ax1[1].set_ylim(0, 1)
    
    # stabillity index
    ax1[2].plot(JCs, stbData2['stb1'], c='magenta')
    ax1[2].plot(JCs, stbData2['stb2'], c='magenta')
    ax1[2].plot(JCs, stbData2['stb3'], c='magenta')
    ax1[2].plot(JCs, stbData2['stb4'], c='magenta')
    ax1[2].plot(JCs, stbData2['stb5'], c='magenta')
    ax1[2].plot(JCs, stbData2['stb6'], c='magenta')
    
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















    # # jacobi constant of initial orbit from linear approximation
    # JC = jacobi_constant(mu, VertIC)
    
    # # get state and period for next orbit in family by perturbing in JC
    # JC = JC+dJC
    # VertIC, P, STM = xzplane_JCfixed(VertIC, P, JC, return_STM=True)
    
    # # transform half period STM to get monodromy matrix
    # Mon = G@np.linalg.inv(STM)@G@STM
    
    # # keep record of states periods and JC for each orbit
    # states = [VertIC]
    # times  = [P]
    # JCs    = [JC]
    
    # # get eigenvalues and eigenvectors
    # vals, vecs = np.linalg.eig(Mon)
    
    # # store history of monodromy matrix, eigvals, stability index, and time constant
    # MonMs, unsorted_vals, unsorted_vecs = [Mon], [vals], [vecs]
    
    # # step through family and get corrected IC for each orbit
    # for i in range(num_steps):
    #     # print something to show progress
    #     if i%20 == 0:
    #         print('currently on step {}/{} of L{} family'.format(i, num_steps, Lpoint_indx+1))
        
    #     # get state and period for next orbit in family by perturbing in JC
    #     newJC = JC+dJC
    #     VertIC, P, STM = xzplane_JCfixed(VertIC, P, newJC, return_STM=True)
        
        
    #     # jacobi constant at current orbit
    #     JC = jacobi_constant(mu, VertIC)
        
    #     # transform half period STM to get monodromy matrix
    #     Mon = G@J@STM.T@np.linalg.inv(J)@G@STM
        
    #     # get eigenvalues and eigenvectors
    #     vals, vecs = np.linalg.eig(Mon)
                             
    #     # add new values to history list
    #     states.append(VertIC)
    #     times.append(P)
    #     JCs.append(JC)
    #     MonMs.append(Mon)
    #     unsorted_vals.append(vals)
    #     unsorted_vecs.append(vecs)
    
    
    
    
    
    
    
    
    
    