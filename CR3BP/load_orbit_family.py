# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:34:33 2023

@author: sam
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
from tools import normalize, sort_eigenvalues, DataFrame_str2array
from targeters import xzplane_xfixed, xzplane_zfixed
from plotting_tools import plot_family_2D



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
    
    
    # path to folder to save data
    filepath = r'C:\Users\sam\Desktop\Research\CR3BP\OrbitFamilies\EarthMoon\Lyapunovs'
    filename = r'\L1Lyapunovs.xlsx'
    
    # linewidth for plotting
    lw = 1
    # color for plotting
    color = 'purple'
    
    # extract previously computed orbit family to a DF
    familyData= pd.read_excel(filepath + filename)
    
    # index the data
    times = familyData['P']
    MonMs = familyData['monodromy matrix'].apply(DataFrame_str2array)
    ICs = familyData['initial condition'].apply(DataFrame_str2array)
    JCs = familyData['JC']
    
    #%%
    # plot the family and get final states
    fig, ax = plt.subplots()
    FCs = plot_family_2D(fig, ax, ICs, times, JCs, mod=1, lw=0.8,
                      cbar_axes = [0.65, 0.11, 0.02, 0.76],
                      xaxis_symmetry=True, 
                      return_final_state=True,
                      return_monodromy=False)
    
    
    #%%
    # compute analytical eigenvalues for comparison
    comp_vals, comp_vecs = [], []
    calc_vals, calc_vecs = [], []
    
    trace, trace1, trace2 = [], [], []
    det, det1, det2 = [], [] ,[]
    
    # now that we have all the monodromy matricies we can iterate through them
    for i in range(len(MonMs)):
        
        # period of current orbit
        t = times[i]*2 
        
        # monodromy matrix of current orbit
        monod = MonMs[i]
        
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
        
        # eigenvalues and vectors from numpy linalg module
        vals1, vecs1 = np.linalg.eig(monod)
        # eigenvalues from analytic method
        vals2 = [lam1, lam2, lam3, lam4, lam5, lam6]
        
        # add all values to their respective lists
        calc_vals.append(vals1)
        comp_vals.append(vals2)
        
        # get trace of monodromy matrix
        trace.append(np.trace(monod))
        # calculate trace as the sum of the eigenvalues
        trace1.append(sum(vals1))
        trace2.append(sum(vals2))
        
        # get determinant of monodromy matrix
        det.append(np.linalg.det(monod))
        # get determinant as product of eigenvalues
        det1.append(np.prod(vals1))
        det2.append(np.prod(vals2))
    
    #%%
    # want to compare the accuracy of the eigenvlaue computation methods
    # the trace of the monodromy matrix should be equal to the sum of the eigenavlues
    # while the determinant should be equal to the product
    
    # compute the difference and see which ones have the smallest error
    diff_trace1 = [trace[i]-trace1[i] for i in range(len(trace))]
    diff_trace2 = [trace[i]-trace2[i] for i in range(len(trace))]
    
    diff_det1 = [det[i]-det1[i] for i in range(len(det))]
    diff_det2 = [det[i]-det2[i] for i in range(len(det))]
    
    #%% 
    # plot difference in trace
    fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 9))
    fig1.set_tight_layout(True)
    
    # plot title
    ax1[0].set_title('Difference Between Trace of Monodromy Matrix and Sum of Eigenvalues')
    
    ax1[0].plot(diff_trace1, c=color, linewidth=lw)
    ax1[1].plot(diff_trace2, c=color, linewidth=lw)
    
    # label for each y axis
    ax1[0].set_ylabel('Numpy Linalg')
    ax1[1].set_ylabel('Analytical')
    
    # only want x label on bottom axis
    ax1[1].set_xlabel('Family Member Index')
    
    #%% 
    # plot difference in determinant
    fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 9))
    fig2.set_tight_layout(True)
    
    # plot title
    ax2[0].set_title('Difference Between Determinant of Monodromy Matrix and Product of Eigenvalues')
    
    ax2[0].plot(diff_det1, c=color, linewidth=lw)
    ax2[1].plot(diff_det2, c=color, linewidth=lw)
    
    # label for each y axis
    ax2[0].set_ylabel('Numpy Linalg')
    ax2[1].set_ylabel('Analytical')
    
    # only want x label on bottom axis
    ax2[1].set_xlabel('Family Member Index')
    
    #%% 
    # plot each determinant
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8, 4))
    fig3.set_tight_layout(True)
    
    # plot title
    ax3.set_title('Difference Between Determinant of Monodromy Matrix and Product of Eigenvalues')
    
    ax3.plot(det, c='purple', linewidth=lw)
    ax3.plot(det1, c='magenta', linewidth=lw)
    ax3.plot(det2, c='cyan', linewidth=lw)
    
    ax3.legend(['Determinant', 'Product of $\lambda$ (Numpy)', 'Product of $\lambda$ (Analytic)'])
    
    # only want x label on bottom axis
    ax3.set_xlabel('Family Member Index')
    
    #%% 
    # initial state JC and final state JC
    fig4, ax4 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 9))
    fig4.set_tight_layout(True)
    
    initJCs  = JCs
    finalJCs = [jacobi_constant(mu, i) for i in FCs]
    diff_JCs = [initJCs[i]-finalJCs[i] for i in range(len(initJCs))]
    
    # plot title
    ax4[0].set_title('Difference Between JC at Initial and Final States')
    
    ax4[0].plot( initJCs, c='purple', linewidth=lw)
    ax4[0].plot(finalJCs, c='magenta', linewidth=lw)
    
    ax4[1].plot(diff_JCs, c=color, linewidth=lw)
    
    # legend
    ax4[0].legend(['initial JC', 'final JC'])
    
    # label for each y axis
    ax4[0].set_ylabel('Jacobi Constant')
    ax4[1].set_ylabel('Difference between each JC')
    
    # only want x label on bottom axis
    ax4[1].set_xlabel('Family Member Index')
    
    
    
    
    
    
    
    
    
    
    
    