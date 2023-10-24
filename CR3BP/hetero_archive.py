# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:39:11 2023
@author: sam


heteroclinic connection targeting schemes that dont work
"""

#%% ALL FORWARD TIME PROPAGATION
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


#%%
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

# propagate backwards from state10 to get state11
sol10 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_s2), state10, 
                dense_output=True, rtol=tol, atol=tol)
state11 = np.array([sol10.y[i][-1] for i in range(6)])

# propagate backwards from state8 to get state9
sol8 = solve_ivp(lambda t, y: EOM(t, y, mu), (t0, -T_s1), state8, 
                dense_output=True, rtol=tol, atol=tol)
state9 = np.array([sol8.y[i][-1] for i in range(6)])  


#%%


# -------------------------------------------------------------------------
# free variable vector
# -------------------------------------------------------------------------

X = np.concatenate((state4, state6, state8, state10, [tau_1], [tau_2],
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
    r8  = X[12:18]
    r10 = X[18:24]
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
    sol8 = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, -Ts1), 
                     np.concatenate([r8, phi0[0]]), 
                     dense_output=True, rtol=tol, atol=tol)
    sol9 = np.array([sol8.y[i][-1] for i in range(len(sol8.y))])
    r9   = sol9[0:6]
    STM9 = sol9[6::].reshape((6, 6))
    
    
    # integrate along second stable manifold segment
    sol10 = solve_ivp(lambda t, y: EOM_STM(t, y, mu), (t0, -Ts2),
                      np.concatenate([r10, phi0[0]]), 
                     dense_output=True, rtol=tol, atol=tol)
    sol11 = np.array([sol10.y[i][-1] for i in range(len(sol10.y))])
    r11   = sol11[0:6]
    STM11 = sol11[6::].reshape((6, 6))
    
    
    # integrate along second orbit segment
    st_guess = get_manifold_ICs(mu, state12, P_L2*2, stepoff, 3, 
                                positive_dir=False, stable=True, 
                                tau_alpha=t2,
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
    F = np.concatenate((r4-r3, r6-r5, r8-r7, r10-r9, r14-r11), axis=0)
    f1.append(np.linalg.norm( r4 -  r3))
    f2.append(np.linalg.norm( r6 -  r5))
    f3.append(np.linalg.norm( r8 -  r7))
    f4.append(np.linalg.norm(r10 -  r9))
    f5.append(np.linalg.norm(r14 - r11))
    
    # -------------------------------------------------------------------------
    # jacobian - partial derivative of F wrt X
    # -------------------------------------------------------------------------
    
    # time derivatives of each state, plus STM for orbit segments
    dr2   = EOM_STM(0, np.concatenate([r2, STM2.reshape(1, 36)[0]]), mu)
    dSTM2 = dr2[6::].reshape((6, 6))
    dr2   = dr2[0:6]
    
    dr5  = EOM(0,  r5, mu)
    dr7  = EOM(0,  r7, mu)
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
    
    row3 = np.concatenate((Z6, -STM7, I6, Z6, 
                           Z1, Z1, Z1, -dr7.reshape(6, 1), Z1, Z1), 
                           axis=1)
    
    row4 = np.concatenate((Z6, Z6, -STM9, I6, 
                           Z1, Z1, Z1, Z1, -dr9.reshape(6, 1), Z1), 
                           axis=1)
    
    row5 = np.concatenate((Z6, Z6, Z6, -STM11, 
                           Z1, dr14dtau.reshape(6, 1), Z1, Z1, Z1, -dr11.reshape(6, 1)), 
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
    
    
    
    
    
    
    
    #%% MIXED TIME PROPAGATION
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


    #%%
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
    
    
    #%%
    
    
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
        F = np.concatenate((r4-r3, r6-r5, r8-r7, r9-r10, r11-r14), axis=0)
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
                               Z1, Z1, Z1, Z1, Z1, dr11.reshape(6, 1)), 
                               axis=1)
        
        row5 = np.concatenate((Z6, Z6, Z6, I6, 
                               Z1, -dr14dtau.reshape(6, 1), Z1, Z1, Z1, Z1), 
                               axis=1)
        
        DF = np.vstack((row1, row2, row3, row4, row5))
        
        update = np.linalg.pinv(DF)@F
        # update = DF.T@(DF@DF.T)**(-1)@F
        # update = DF.T@np.linalg.pinv(DF@DF.T)@F
        
        # update the free variable vector
        Xnew = X - update
        iterations += 1
        error = np.linalg.norm(F)
        error_history.append(error)
        
        if iterations > 10:
            print('iteration max reached, error = {}'.format(error))
            break