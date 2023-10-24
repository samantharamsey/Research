# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:09:46 2023
@author: sam

get halo orbit families
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
from CR3BP_tools import make_family_data
from tools import normalize, sort_eigenvalues
from targeters import xzplane_xfixed, xzplane_zfixed
from plotting_tools import plot_family_3D, plot_primaries_3D, plot_orbit_3D, plot_Lpoints_3D
    
        

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
    
    # define the libration point to compute family at
    L1 = True
    L2 = False
    L3 = False
    
    # matricies for transforming the half period STM to the monodromy matrix
    # based on symmetry of desired family
    G = np.array([[ 1,  0,  0,  0,  0,  0],
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
    filepath = r'C:\Users\sam\Desktop\Research\CR3BP\OrbitFamilies\EarthMoon\Halos'
    resave = False
    
    if L1==True:
        
        # libration point of interest (index 0-4)
        Lpoint_indx = 0
        
        # determine step size in x and z
        dx = 0.0001
        dz = 0.0005
        
        # lower and upper bounds for when to step in z
        zlow  = 0.12
        zhigh = 0.24
        
        # max values to terminate loop
        xmax  = 0.00
        zmax  = 0.35
        
        # terminate targeting integration when going from positive to negative boolean
        pos2neg = True
        
        # total number of steps to take
        num_steps = 300
        
        # name of final data file for excel export
        filename = r'\L1Halos.xlsx'
        
        # libration point of interest
        Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
        
        # initial conditions for the lyapunov that bifurcates to the halos (grebow 2006 pg 40)
        LyapIC_L1 = np.array([0.8234, 0.0, 0.0, 0.0, 0.1263, 0.0])
        HaloIC_L1, P_L1, STM_L1 = xzplane_xfixed(LyapIC_L1, pos2neg=pos2neg, return_STM=True)
        
        # transform half period STM to get monodromy matrix
        Mon_L1 = G@J@STM_L1.T@np.linalg.inv(J)@G@STM_L1
        
        # redefine based on chosen starting point
        HaloIC, P, Mon = HaloIC_L1, P_L1, Mon_L1
    
    if L2==True:
        
        # libration point of interest (index 0-4)
        Lpoint_indx = 1
        
        # determine step size in x and z
        dx = -0.0001
        dz =  0.0005
        
        # lower and upper bounds for when to step in z
        zlow  = 0.03
        zhigh = 0.99
        
        # max values to terminate loop
        xmax  = 0.993
        zmax  = 0.99
        
        # terminate targeting integration when going from positive to negative boolean
        pos2neg = False
        
        # total number of steps to take
        num_steps = 4000
        
        # name of final data file for excel export
        filename = r'\L2Halos.xlsx'
        
        # libration point of interest
        Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
        
        # initial conditions for the lyapunov that bifurcates to the halos (grebow 2006 pg 40)
        LyapIC_L2 = np.array([1.1809, 0.0, 0.0, 0.0, -0.1559, 0.0])
        HaloIC_L2, P_L2, STM_L2 = xzplane_xfixed(LyapIC_L2, return_STM=True)
        
        # transform half period STM to get monodromy matrix
        Mon_L2 = G@J@STM_L2.T@np.linalg.inv(J)@G@STM_L2
        
        # redefine based on chosen starting point
        HaloIC, P, Mon = HaloIC_L2, P_L2, Mon_L2
        
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
        dx =  0.0001
        dz = -0.0005
        
        # lower and upper bounds for when to step in z
        zlow  = 0.03
        zhigh = 0.08
        
        # max values to terminate loop
        xmax  = -1
        zmax  = 0.99
        
        # terminate targeting integration when going from positive to negative boolean
        pos2neg = True
        
        # total number of steps to take
        num_steps = 400
        
        # name of final data file for excel export
        filename = r'\L3Halos.xlsx'
        
        # libration point of interest
        Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
        
        # initial conditions for the lyapunov that bifurcates to the halos (grebow 2006 pg 40)
        LyapIC_L3 = np.array([-1.6967, 0.0, 0.0, 0.0, 1.2796, 0.0])
        HaloIC, P = xzplane_xfixed(LyapIC_L3, pos2neg=pos2neg)
        
        # 3D plot formatting
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection= '3d')
        HaloFC_L2 = plot_orbit_3D(ax, HaloIC, P, 'black', return_final_state=True)
        
        # recorrect based on chosen starting point
        pos2neg = False
        HaloIC, P, STM_L3 = xzplane_xfixed(HaloFC_L2, pos2neg=pos2neg, return_STM=True)
        
        # transform half period STM to get monodromy matrix
        Mon = G@J@STM_L3.T@np.linalg.inv(J)@G@STM_L3
        
        
    #%% 
    #NATURAL PARAMETER CONTINUATION FOR HALO FAMILY
    
    # jacobi constant of initial orbit from linear approximation
    JC = jacobi_constant(mu, HaloIC)
    
    # save previous step to get slope for initial guess at new step
    prevIC = HaloIC
    prevP  = P
    
    # get state and period for next orbit in family by perturbing in z
    currIC = np.array([HaloIC[0], 0, HaloIC[2]+dz, 0, HaloIC[4], 0])
    currIC, currP, currSTM = xzplane_zfixed(currIC, pos2neg=pos2neg, return_STM=True)
    currJC = jacobi_constant(mu, currIC)
    
    # transform half period STM to get monodromy matrix
    currMon = G@J@currSTM.T@np.linalg.inv(J)@G@currSTM
    
    # keep record of states periods and JC for each orbit
    states = [currIC]
    times  = [currP]
    JCs    = [currJC]
    
    # get eigenvalues and eigenvectors
    vals1, vecs1 = np.linalg.eig(Mon)
    vals2, vecs2 = np.linalg.eig(currMon)
    
    # store history of monodromy matrix, eigvals, stability index, and time constant
    MonMs, unsorted_vals, unsorted_vecs = [currMon], [vals2], [vecs2]
    
    # step through family and get corrected IC for each orbit
    for i in range(num_steps):
        # print something to show progress
        if i%20 == 0:
            print('currently on step {}/{} of L{} family'.format(i, num_steps, Lpoint_indx+1))
        
        # state components for current and previous orbits
        x1, y1, z1, vx1, vy1, vz1 = prevIC
        x2, y2, z2, vx2, vy2, vz2 = currIC
        
        # initialize both to false
        zfixed = False
        xfixed = False
        
        # x-fixed or z-fixed targeting
        if abs(z2) < zlow or abs(z2) > zhigh:
            zfixed=True
        else:
            xfixed=True
            
        # break out of for loop if z > zmax
        if z2 > zmax or x2 < xmax:
            break
            
        if zfixed:
            # vy0 slope wrt z0
            mz = (vy2-vy1)/(z2-z1)
            bz = vy1 - mz*z1
            
            # step in z
            z3 = z2 + dz
            # new vy from slope of line
            vy3 = mz*z3 + bz
            
            state1 = np.array([x2, y2, z3, vx2, vy3, vz2])
            HaloIC, P, STM = xzplane_zfixed(state1, pos2neg=pos2neg, return_STM=True)
            
        if xfixed:
            # vy0 slope wrt x0
            mx = (vy2-vy1)/(x2-x1)
            bx = vy1 - mx*x1
            
            # step in x
            x3 = x2 + dx
            # new vy from slope of line
            vy3 = mx*x3 + bx
            
            state1 = np.array([x3, y2, z2, vx2, vy3, vz2])
            HaloIC, P, STM = xzplane_xfixed(state1, pos2neg=pos2neg, return_STM=True) 
        
        
        # jacobi constant at current orbit
        JC = jacobi_constant(mu, HaloIC)
        
        # transform half period STM to get monodromy matrix
        Mon = G@J@STM.T@np.linalg.inv(J)@G@STM
        
        # get eigenvalues and eigenvectors
        vals, vecs = np.linalg.eig(Mon)
                             
        # add new values to history list
        states.append(HaloIC)
        times.append(P)
        JCs.append(JC)
        MonMs.append(Mon)
        unsorted_vals.append(vals)
        unsorted_vecs.append(vecs)
        
        prevIC = currIC
        currIC = HaloIC
        
    #%%
    # sort the eigenvalues and eigenvectors
    sorted_vals, sorted_vecs = sort_eigenvalues(unsorted_vals, unsorted_vecs)
        
    #%%
    # plot the family
        
    # set title
    title = 'Earth-Moon L$_{}$ Halo Family'.format(Lpoint_indx+1)
    
    # 3D plot formatting
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection= '3d')
    
    # plot formatting
    ax.set_title(title)
    
    # plot the libration point locations
    plot_Lpoints_3D(ax, L1=pltL1, L2=pltL2, L3=pltL3, L4=pltL4, L5=pltL5)
    
    # plot locations of the primaries
    plot_primaries_3D(ax, primary=pltPri, secondary=pltSec, alpha=0.5,
                       primary_color='mediumseagreen',
                       secondary_color='gray')

    # plot the family of orbits
    plot_family_3D(fig, ax, states, times, JCs, mod=10, lw=0.8,
                      xzplane_symmetry=True)

    
    #%%
    # send all data to a DataFrame
    FamilyData = make_family_data(states, times, JCs, MonMs, sorted_vals, sorted_vecs,
                                  plot_results=True)
    
    #%%
    # send DataFrame to an excel file for storage
    if resave==True:
        FamilyData.to_excel(filepath + filename)
    

    

    
    

    