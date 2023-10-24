# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:52:55 2023

@author: sam
"""

import numpy as np
import itertools
import pickle
import sys
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt

# load in constants
pickle_off = open("general_params.pickle", 'rb')
general_params = pickle.load(pickle_off)
# system specific values
pickle_off = open("system_data.pickle", 'rb')
system_data = pickle.load(pickle_off)
    
# define constants
tol = general_params['tol']

# start and end times for propagation incase no event detected
t0 = general_params['t0']
tf = general_params['tf']

# system mass parameter
mu = system_data['mu']
# step off distance
stepoff = system_data['stepoff']

# append CR3BP folder to path so we can call functions from it
path2CR3BP = general_params['path2CR3BP']
sys.path.append(path2CR3BP)

from CR3BP_equations import EOM, EOM_2D, EOM_STM, EOM_STM_2D, jacobi_constant, libsolver

def plot_Lpoints_3D(ax, L1=True, L2=True, L3=True, L4=True, L5=True):
    # determine the locations of the libration points
    xpoints, ypoints = libsolver(mu, tol) 
    
    # plot the libration point locations
    if L1==True:
        ax.scatter(xpoints[0], ypoints[0], 0, zorder=3, color = 'goldenrod', marker = '*')
    if L2==True:
        ax.scatter(xpoints[1], ypoints[1], 0, zorder=3, color = 'goldenrod', marker = '*')
    if L3==True:
        ax.scatter(xpoints[2], ypoints[2], 0, zorder=3, color = 'goldenrod', marker = '*')
    if L4==True:
        ax.scatter(xpoints[3], ypoints[3], 0, zorder=3, color = 'goldenrod', marker = '*')
    if L5==True:
        ax.scatter(xpoints[4], ypoints[4], 0, zorder=3, color = 'goldenrod', marker = '*')
    
    

def plot_family_2D(fig, ax, orbitICs, times, JCs, mod=1, lw=0.8,
                  cbar_axes = [0.65, 0.11, 0.02, 0.76],
                  xaxis_symmetry=False, 
                  return_final_state=False,
                  return_monodromy=False):
    '''
    plots a family of periodic orbits in the xy plane colored by JC
    given lists of initial conditions, periods, and JCs
    '''
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(6)
    phi0 = phi0.reshape(1, 36)
    
    # Jacobi Constant color mapping
    cmap = plt.cm.cool
    norm = plt.Normalize(vmin=np.min(JCs), vmax=np.max(JCs))
    cbar_ax = fig.add_axes(cbar_axes)
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, label='Jacobi Constant', 
                                    norm=norm, orientation='vertical') 
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    ax.set_xlabel('x (nd)')
    ax.set_ylabel('y (nd)')
    
    FCs = []
    
    # iterate through each orbit in list
    for i in range(len(orbitICs)):
        # only plot every 20 orbits so theyre spaced out to visualize
        if i%mod == 0:
        
            # extract current states, times, and JC
            tf = times[i]
            JC = jacobi_constant(mu, orbitICs[i])
            
            # integrate the orbit for the full period
            if return_monodromy==True:
                state = np.concatenate([orbitICs[i], phi0[0]])
                sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state,
                                dense_output = True, rtol = tol, atol = tol)
            else:
                state = orbitICs[i]
                sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tf), state, 
                                dense_output=True, rtol=tol, atol=tol)
            
            # plot orbit color mapped to JC
            ax.plot(sol.y[0], sol.y[1], color=cmap(norm(JC)), linewidth=lw)
            
                    
            # plot orbit symmetric about x-axis
            if xaxis_symmetry==True:
                ax.plot(sol.y[0], -sol.y[1], color=cmap(norm(JC)), linewidth=lw)
            
            if return_final_state==True:
                FCs.append(np.array([sol.y[i][-1] for i in range(len(sol.y))]))
        
    # return final integrated state
    if return_final_state==True:
        return FCs
    
def plot_family_3D(fig, ax, orbitICs, times, JCs, mod=1, lw=0.8,
                  cbar_axes = [0.65, 0.11, 0.02, 0.76], 
                  xyplane_symmetry=False,
                  yzplane_symmetry=False,
                  xzplane_symmetry=False, 
                  return_final_state=False,
                  return_monodromy=False):
    '''
    plots a family of periodic orbits in 3D colored by JC
    given lists of initial conditions, periods, and JCs
    '''
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(6)
    phi0 = phi0.reshape(1, 36)
    
    # Jacobi Constant color mapping
    cmap = plt.cm.cool
    norm = plt.Normalize(vmin=np.min(JCs), vmax=np.max(JCs))
    cbar_ax = fig.add_axes(cbar_axes)
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, label='Jacobi Constant', 
                                    norm=norm, orientation='vertical') 
    
    # add a padding factor for the axis labels
    pad=10
    ax.set_xlabel('x (nd)', labelpad=pad)
    ax.set_ylabel('y (nd)', labelpad=pad)
    ax.set_zlabel('z (nd)', labelpad=pad)
    
    FCs = []
    
    # iterate through each orbit in list
    for i in range(len(orbitICs)):
        # only plot every 20 orbits so theyre spaced out to visualize
        if i%mod == 0:
        
            # extract current states, times, and JC
            tf = times[i]
            JC = jacobi_constant(mu, orbitICs[i])
            
            # integrate the orbit for the full period
            if return_monodromy==True:
                state = np.concatenate([orbitICs[i], phi0[0]])
                sol = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, tf), state,
                                dense_output = True, rtol = tol, atol = tol)
            else:
                state = orbitICs[i]
                sol = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, tf), state, 
                                dense_output=True, rtol=tol, atol=tol)
            
            # plot orbit color mapped to JC
            ax.plot(sol.y[0], sol.y[1], sol.y[2], color=cmap(norm(JC)), linewidth=lw)
            
            # plot orbit symmetric about planes
            if xyplane_symmetry==True:
                ax.plot(-sol.y[0], -sol.y[1],  sol.y[2], color=cmap(norm(JC)), linewidth=lw)
            if yzplane_symmetry==True:
                ax.plot(-sol.y[0],  sol.y[1],  sol.y[2], color=cmap(norm(JC)), linewidth=lw)
            if xzplane_symmetry==True:
                ax.plot( sol.y[0], -sol.y[1],  sol.y[2], color=cmap(norm(JC)), linewidth=lw)
            
            if return_final_state==True:
                FCs.append(np.array([sol.y[i][-1] for i in range(len(sol.y))]))
    
    # formatting
    set_axes_equal(ax)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # return final integrated state
    if return_final_state==True:
        return FCs
    
def plot_primaries_3D(ax, primary=True, secondary=True, alpha=0.5,
                   primary_color='mediumseagreen',
                   secondary_color='gray'):
    '''
    plots to scale 3D spheres for one or both primary bodies
    '''
    
    # list of angles
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0,   np.pi, 100)
    
    # plot the larger primary location
    if primary==True:
        # 3D earth
        r = system_data['r primary']
        
        # sphere math
        x = r*np.outer(np.cos(u), np.sin(v)) + (-mu)
        y = r*np.outer(np.sin(u), np.sin(v))
        z = r*np.outer(np.ones(np.size(u)), np.cos(v))
        
        # plot the surface
        ax.plot_surface(x, y, z, color=primary_color, alpha=alpha)
    
    # plot the smaller primary location
    if secondary==True:
        # 3D moon
        r = system_data['r secondary']
        
        # sphere math
        x = r*np.outer(np.cos(u), np.sin(v)) + (1-mu)
        y = r*np.outer(np.sin(u), np.sin(v))
        z = r*np.outer(np.ones(np.size(u)), np.cos(v))
        
        # plot the surface
        ax.plot_surface(x, y, z, color=secondary_color, alpha=alpha)
    
    
def plot_orbit_2D(ax, orbitIC, P, color, 
                  xaxis_symmetry=False, 
                  return_final_state=False,
                  return_monodromy=False):
    '''
    plots an orbit in the xy plane given the initial condtions and period
    '''
    
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
    
def plot_orbit_3D(ax, orbitIC, P, color, 
                  xyplane_symmetry=False,
                  yzplane_symmetry=False,
                  xzplane_symmetry=False,
                  return_final_state=False):
    '''
    plots a 3D orbit given the initial condtions and period
    '''
    
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
    
    # formatting
    set_axes_equal(ax)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # return final integrated state
    if return_final_state==True:
        return np.array([sol.y[i][-1] for i in range(6)])
    
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
