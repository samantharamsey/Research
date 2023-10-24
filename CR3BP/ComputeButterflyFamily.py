# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:09:46 2023
@author: sam

get butterfly orbit families
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import os


import pickle
# load in constants
pickle_off = open('general_params.pickle', 'rb')
general_params = pickle.load(pickle_off)
# Earth-Moon specific values
pickle_off = open('system_data.pickle', 'rb')
system_data = pickle.load(pickle_off)

import sys
# append CR3BP folder to path so we can call functions from it
path2CR3BP = general_params['path2CR3BP']
sys.path.append(path2CR3BP)

from CR3BP_equations import libsolver, jacobi_constant, EOM, EOM_STM, pseudo_potential
from CR3BP_tools import make_family_data
from tools import normalize, sort_eigenvalues
from targeters import xzplane_tfixed
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
    filepath = r'C:\Users\sam\Desktop\Research\CR3BP\OrbitFamilies\EarthMoon\Butterflies'
    resave = True
    
    # libration point of interest (index 0-4)
    Lpoint_indx = 1
    
    # determine step size in period
    dP = 0.001
    
    # terminate targeting integration when going from positive to negative boolean
    pos2neg = False
    
    # total number of steps to take
    num_steps = 5000
    
    # name of final data file for excel export
    filename = r'\L2Butterflies.xlsx'
    
    # libration point of interest
    Lpoint = [xpoints[Lpoint_indx], ypoints[Lpoint_indx], 0, 0, 0, 0]
    
    # initial conditions for the halo that bifurcates to the butterflies (grebow 2006 pg 74)
    HaloIC_L2 = np.array([1.011858013, 0.0, 0.173956429, 0.0, -0.070011235, 0.0])
    # single period Halo 
    P1 = 0.68737349*2
    # period doubling Halo
    P2 = P1*2
    
    # correct to a butterfly orbit with twice the period
    ButtIC, STM = xzplane_tfixed(HaloIC_L2, P2, return_STM=True)
    
    # redefine based on chosen starting point
    OrbitIC, P, Mon = ButtIC, P2, STM
        
        
        
    #%% 
    # NATURAL PARAMETER CONTINUATION FOR Butterfly FAMILY
    
    # jacobi constant of initial orbit from linear approximation
    JC = jacobi_constant(mu, OrbitIC)
    
    # get eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(Mon)
    
    # keep record of states periods and JC for each orbit
    states = [OrbitIC]
    times  = [P]
    JCs    = [JC]
    
    # store history of monodromy matrix, eigvals, stability index, and time constant
    MonMs, unsorted_vals, unsorted_vecs = [Mon], [vals], [vecs]
    
    # step through family and get corrected IC for each orbit
    for i in range(num_steps):
        # print something to show progress
        if i%10 == 0:
            print('currently on step {}/{} of L{} family'.format(i, num_steps, Lpoint_indx+1))
        
        P2 += dP
        OrbitIC, STM = xzplane_tfixed(OrbitIC, P2, return_STM=True)
        
        # jacobi constant at current orbit
        JC = jacobi_constant(mu, OrbitIC)
        
        # transform half period STM to get monodromy matrix
        Mon = STM
        
        # get eigenvalues and eigenvectors
        vals, vecs = np.linalg.eig(Mon)
                             
        # add new values to history list
        states.append(OrbitIC)
        times.append(P2)
        JCs.append(JC)
        MonMs.append(Mon)
        unsorted_vals.append(vals)
        unsorted_vecs.append(vecs)
        
    #%%
    # sort the eigenvalues and eigenvectors
    sorted_vals, sorted_vecs = sort_eigenvalues(unsorted_vals, unsorted_vecs)
        
    #%%
    # plot the family
        
    # set title
    title = 'Earth-Moon L$_{}$ Butterfly Family'.format(Lpoint_indx+1)
    
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
    plot_family_3D(fig, ax, states, times, JCs, mod=100, lw=0.8,
                   cbar_axes = [0.8, 0.11, 0.02, 0.76],
                      xzplane_symmetry=True)

    
    #%%
    # send all data to a DataFrame
    FamilyData = make_family_data(states, times, JCs, MonMs, sorted_vals, sorted_vecs,
                                  plot_results=True)
    
    #%%
    # send DataFrame to an excel file for storage
    if resave==True:
        FamilyData.to_excel(filepath + filename)
    

    

    
    

    