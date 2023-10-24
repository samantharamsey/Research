# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:04:27 2023
@author: samantha ramsey

data and constants for Earth-Moon system
"""

import pickle


# -----------------------------------------------------------------------------
# general parameters
# -----------------------------------------------------------------------------

general_params = {'t0':    0.0,
                  'tf':    6,
                  'tol':   1e-12,
                  'path2CR3BP': r'C:\Users\sam\Desktop\Research\CR3BP'}

pickling_on = open("general_params.pickle","wb")
pickle.dump(general_params, pickling_on)
pickling_on.close()



system = 'Earth-Moon'


# -----------------------------------------------------------------------------
# specific system data
# -----------------------------------------------------------------------------

# Earth-Moon systems
if system == 'Earth-Moon':
    lstar = 384400
    system_data = {'mu':    0.012153659, 
                   'lstar': lstar,
                   'tstar': 3.751903e5,
                   'stepoff': 50/lstar,
                   'r secondary':  1737.4/lstar,
                   'r primary': 6378.1/lstar}
    
# Nadia's values
if system == 'Nadia-Earth-Moon':
    lstar = 384400
    system_data = {'mu':    0.012150586550568, 
                   'lstar': lstar,
                   'tstar': 3.751903e5,
                   'stepoff': 50/lstar,
                   'r secondary':  1737.4/lstar,
                   'r primary': 6378.1/lstar}
    
# Liam's values
if system == 'Liam-Earth-Moon':
    lstar = 384400
    system_data = {'mu':    0.01215361, 
                   'lstar': lstar,
                   'tstar': 3.751903e5,
                   'stepoff': 50/lstar,
                   'r secondary':  1737.4/lstar,
                   'r primary': 6378.1/lstar}
    
# Sun-Earth system
if system == 'Sun-Earth':
    system_data = {'mu':    0.012153659, 
                   'lstar': 384400,
                   'tstar': 3.751903e5,
                   'stepoff': 25/384400}

pickling_on = open("system_data.pickle","wb")
pickle.dump(system_data, pickling_on)
pickling_on.close()



