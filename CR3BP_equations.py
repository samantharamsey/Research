 # -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:59:08 2022

@author: saman
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def gamma1(g, mu):
    '''
    Equation of motion for Collinear Libration point 1
    '''
    gamma  =   g**5 -   (3-mu)*g**4 +   (3-2*mu)*g**3 -   mu*g**2 + 2*mu*g - mu
    gammap = 5*g**4 - 4*(3-mu)*g**3 + 3*(3-2*mu)*g**2 - 2*mu*g    + 2*mu
    return gamma, gammap


def gamma2(g, mu):
    '''
    Equation of motion for Collinear Libration point 2
    '''
    gamma  =   g**5 +   (3-mu)*g**4 +   (3-2*mu)*g**3 -   mu*g**2 + 2*mu*g - mu
    gammap = 5*g**4 + 4*(3-mu)*g**3 + 3*(3-2*mu)*g**2 - 2*mu*g    + 2*mu
    return gamma, gammap


def gamma3(g, mu):
    '''
    Equation of motion for Collinear Libration point 3
    '''
    gamma  =   g**5 +   (2+mu)*g**4 +   (1 + 2*mu)*g**3 -   (1-mu)*g**2 - 2*(1-mu)*g - (1-mu)
    gammap = 5*g**4 + 4*(2+mu)*g**3 + 3*(1 + 2*mu)*g**2 - 2*(1-mu)*g    - 2*(1-mu)
    return gamma, gammap


def newton_method(function, x0, step_size, tolerance, mu):
    '''
    Determines the roots of a non-linear single variable function using 
    derivative estimation and Taylor Series Expansion
    Args:
        x0 - initial condition estimate
        tolerance - the tolerance for convergence
        step_size - determines the size of delta x
    '''
    
    f, df     = function(x0, mu)
    residual  = abs(x0)
    iteration = 1
    
    while residual > tolerance:
        x1        = x0 - f/df
        f1, df1   = function(x1, mu)
        residual  = abs(x1 - x0)
        f, df     = f1, df1
        x0        = x1
        iteration = iteration + 1
        
    return x1


def libsolver(mu, tol):
    '''
    Solves for the Libration points for a given system
    '''
    
    # solve for the first collinear point (between the primaries)
    g1   = (mu/(3*(1 - mu)))**(1/3)
    g1   = newton_method(gamma1, g1, 1, tol, mu)
    x1   = 1 - mu - g1
    
    # solve for the second collinear point (behind the second primary)
    g2   = (mu/(3*(1 - mu)))**(1/3)
    g2   = newton_method(gamma2, g2, 1, tol, mu)
    x2   = 1 - mu + g2
    
    # solve for the third collinear point (behind the first primary)
    g3   = (-7/12)*mu + 1
    g3   = newton_method(gamma3, g3, 1, tol, mu)
    x3   = -mu - g3
    
    x4   = (1/2) - mu
    x5   = (1/2) - mu
    
    xvec = [x1, x2, x3, x4, x5]
    yvec = [0, 0, 0, np.sqrt(3)/2, -np.sqrt(3)/2]
    
    return xvec, yvec

def pseudo_potential(mu, r):
    '''
    pseudo potential and its partial derivatives
    '''
    x, y, z = r[0], r[1], r[2]
    
    d = np.sqrt((x + mu    )**2 + y**2 + z**2)
    r = np.sqrt((x + mu - 1)**2 + y**2 + z**2)
    
    U  = (x**2 + y**2)/2 + (1 - mu)/d + mu/r
    
    Ux  = x - (1 - mu)*(x + mu)/d**3 - mu*(x + mu - 1)/r**3
    Uy  = y - (1 - mu)*y/d**3        - mu*y/r**3
    Uz  =   - (1 - mu)*z/d**3        - mu*z/r**3
    
    Uxx = 1 - (1 - mu)/d**3 - mu/r**3 + 3*(1 - mu)*(x + mu)**2/d**5 + 3*mu*(x + mu - 1)**2/r**5
    Uyy = 1 - (1 - mu)/d**3 - mu/r**3 + 3*(1 - mu)* y      **2/d**5 + 3*mu* y          **2/r**5
    Uzz =   - (1 - mu)/d**3 - mu/r**3 + 3*(1 - mu)* z      **2/d**5 + 3*mu* z          **2/r**5
    
    Uxy = 3*(1 - mu)*(x + mu)*y/d**5 + 3*mu*(x + mu - 1)*y/r**5
    Uxz = 3*(1 - mu)*(x + mu)*z/d**5 + 3*mu*(x + mu - 1)*z/r**5
    Uyz = 3*(1 - mu)*y*z/d**5        + 3*mu*y*z/r**5
    
    Uyx = Uxy
    Uzx = Uxz
    Uzy = Uyz
    
    return U, Ux, Uy, Uz, Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz

def EOM_STM(t, s, mu):
  '''
  Differentiates the state and STM
  '''
  
  # initialize an empty array for the derivative of f
  F = np.zeros([42])
  
  # unpack the state vector
  x, y, z, dx, dy, dz = s[0], s[1], s[2], s[3], s[4], s[5]
  
  # distance to the primary and secondary
  r1  = np.sqrt((x     + mu)**2 + y**2 + z**2)
  r2  = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
  
  # differential equations of motion
  ddx = x + 2*dy - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
  ddy = y - 2*dx - (1 - mu)*y       /r1**3 - mu*y           /r2**3
  ddz =          - (1 - mu)*z       /r1**3 - mu*z           /r2**3
  
  # differential state vector
  F[0], F[1], F[2], F[3], F[4], F[5]  = dx, dy, dz, ddx, ddy, ddz
  
  # define A matrix (page 242 of Szebehely)
  A11 = 1 - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*(x + mu)**2/r1**5 + 3*mu*(x - 1 + mu)**2/r2**5
  A12 =                                 3*(1 - mu)*(x + mu)*y /r1**5 + 3*mu*(x - 1 + mu)*y /r2**5
  A13 =                                 3*(1 - mu)*(x + mu)*z /r1**5 + 3*mu*(x - 1 + mu)*z /r2**5
  A21 = A12
  A22 = 1 - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*y**2       /r1**5 + 3*mu*y**2           /r2**5
  A23 =                                 3*(1 - mu)*y*z        /r1**5 + 3*mu*y*z            /r2**5
  A31 = A13
  A32 = A23
  A33 =   - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*z**2       /r1**5 + 3*mu*z**2           /r2**5
  
  A      = np.array([[   0,     0,     0,     1,     0,     0],
                     [   0,     0,     0,     0,     1,     0],
                     [   0,     0,     0,     0,     0,     1],
                     [ A11,   A12,   A13,     0,     2,     0],
                     [ A21,   A22,   A23,    -2,     0,     0],
                     [ A31,   A32,   A33,     0,     0,     0]])
  
  # state transition matrix
  phi    = np.array([[ s[6],  s[7],  s[8],  s[9], s[10], s[11]],
                     [s[12], s[13], s[14], s[15], s[16], s[17]],
                     [s[18], s[19], s[20], s[21], s[22], s[23]],
                     [s[24], s[25], s[26], s[27], s[28], s[29]],
                     [s[30], s[31], s[32], s[33], s[34], s[35]],
                     [s[36], s[37], s[38], s[39], s[40], s[41]]])
  
  # update the STM
  STM    = A@phi
  # add STM elements to the state
  F[6::] = STM.reshape(1, 36)
  return F

def EOM(t, s, mu):
  '''
  Differentiates the state without STM
  '''
  
  # unpack the state vector
  x, y, z, dx, dy, dz = s[0], s[1], s[2], s[3], s[4], s[5]
  
  # distance to the primary and secondary
  r1  = np.sqrt((x     + mu)**2 + y**2 + z**2)
  r2  = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
  
  # differential equations of motion
  ddx = x + 2*dy - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
  ddy = y - 2*dx - (1 - mu)*y       /r1**3 - mu*y           /r2**3
  ddz =          - (1 - mu)*z       /r1**3 - mu*z           /r2**3
  
  # differential state vector
  dstate = np.array([dx, dy, dz, ddx, ddy, ddz])
  
  return dstate

def jacobi_constant(mu, s):
    ''' Caluclate the Jacobi constant at a specific position velocity '''
    
    # break the state into the position and velocity components
    x, y, z, vx, vy, vz = s
    
    # define the magnitude of the velocity
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # define the magnitude of the vector to each primary
    d = np.sqrt((x + mu)**2 + y**2 + z**2)
    r = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    
    # calculate the pseudo potential
    U = (1 - mu)/d + mu/r + (1/2)*(x**2 + y**2)
    
    # return the Jacobi Constant
    C = 2*U - v**2
    return C

def nd2d(state, lstar, tstar):
    '''
    converts from dimensional to nondimensional units
    '''
    x = state[0]*lstar
    y = state[1]*lstar
    z = state[2]*lstar
    dx = state[3]*lstar/tstar
    dy = state[4]*lstar/tstar
    dz = state[5]*lstar/tstar
    
    return np.array([x, y, z, dx, dy, dz])

def d2nd(state, lstar, tstar):
    '''
    converts from dimensional to nondimensional units
    '''
    x = state[0]/lstar
    y = state[1]/lstar
    z = state[2]/lstar
    dx = state[3]/(lstar/tstar)
    dy = state[4]/(lstar/tstar)
    dz = state[5]/(lstar/tstar)
    
    return np.array([x, y, z, dx, dy, dz])

