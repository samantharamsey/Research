# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:39:29 2023

@author: sam
"""

def get_manifold_ICs_2D(mu, orbitIC, P, stepoff, num_manifolds, 
                    positive_dir=True, stable=True, tau_alpha=None,
                    return_eigvecs=False, return_STMs=False, return_fixedpoints=False):
    
    if tau_alpha==None:
        # need set of discreetized points around orbit
        tpoints = np.linspace(t0, P, num_manifolds, endpoint=False)
        
    else:
        tpoints = np.array([t0, tau_alpha, P])
    
    # initialize the STM with identity matrix reshaped to an array
    phi0 = np.identity(4)
    phi0 = phi0.reshape(1, 16)
    
    # add STM to state
    orbitIC = np.concatenate([orbitIC, phi0[0]])
    
    # integrate initial conditions for one full period
    sol = solve_ivp(lambda t, y: EOM_STM_2D(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, rtol=tol, atol=tol)
    
    # extract final values from integration results
    phi = np.array([sol.y[i][-1] for i in range(len(sol.y))])
    
    # monodromy matrix
    monod  = phi[4::].reshape((4, 4))
    
    # get eigenvalues nad vectors of monodromy matrix
    vals, vecs = np.linalg.eig(monod)
    
    # create a list of the eigenvalue indicies in ascending order
    idxs = list(range(0, 4))
    idxs.sort(key = lambda x:np.abs(vals[x]))
    
    # stable eigenvalue with be the one with the smallest magnitude
    stble_vec = np.real(vecs[:, idxs[ 0]])
    unstb_vec = np.real(vecs[:, idxs[-1]])
    
    # integrate initial conditions for one full period
    sol = solve_ivp(lambda t, y: EOM_STM_2D(t, y, mu), (t0, P), orbitIC, 
                    dense_output=True, t_eval=tpoints, rtol=tol, atol=tol)
    
    # make a list of states and STMs at each point
    states, STMs = [], []
    for j in range(num_manifolds):
        # extract final values from integration results
        phi = np.array([sol.y[i][j] for i in range(len(sol.y))])
        states.append(phi[0:4])
        STMs.append(phi[4::].reshape((4, 4)))
    
    # list of initial conditions for step off onto manifold
    manifoldIC = []
    # list of stable and unstable eigenvectors
    stvecs, unvecs = [], []
    
    for i, STM in enumerate(STMs):
        
        # make a copy of the state at the current point in the orbit
        state = np.copy(states[i])
        
        # floquet theory to transition the eigenvectors
        st_vec = STM@stble_vec
        un_vec = STM@unstb_vec
        stvecs.append(st_vec)
        unvecs.append(un_vec)
        
        # perturbation from orbit onto stable/unstable eigenvector
        if stable:
            pert = stepoff*(st_vec/np.linalg.norm(st_vec[0:2]))
        else:
            pert = stepoff*(un_vec/np.linalg.norm(un_vec[0:2]))
        
        # positive direction
        if positive_dir:
            if pert[0] > 0:
                state[0:4] = state[0:4] + pert
            else:
                state[0:4] = state[0:4] - pert
        # negative direction
        else:
            if pert[0] < 0:
                state[0:4] = state[0:4] + pert
            else:
                state[0:4] = state[0:4] - pert
        
        manifoldIC.append(state)
    
    returns = [manifoldIC, tpoints]
    
    # add in optional returns
    if return_eigvecs==True:
        returns.append(stvecs)
        returns.append(unvecs)
    if return_STMs==True:
        returns.append(STMs)
    if return_fixedpoints==True:
        returns.append(states)
        
    return returns