# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:58:00 2019

@author: MLempp
"""

import numpy as np
from scipy.integrate import odeint, ode
from scipy.misc import derivative
import matplotlib.pyplot as plt
from random import sample
from numpy import linalg
import sympy as sp
import seaborn as sns
import matplotlib.cm as cm



# =============================================================================
# ode model 
# =============================================================================
def ODEmodel(z,t,par):
    
    m1      = z[0]
    e1      = z[1]
    e2      = z[2]
    
    r0      = par[0]
    kcat1   = par[1]
    kcat2   = par[2]
    Km1     = par[3]
    Km2     = par[4] 
    beta1m  = par[5]
    beta2m  = par[6]
    ind     = par[7]
    kt1     = par[8]
    kt2     = par[9] 
    alpha   = par[10]
    sets    = par[11]  #(b1, b2)

    r1      = kcat1 * e1 * (m1/(m1+Km1))
    r2      = kcat2 * e2 * (m1/(m1+Km2))
    
    mue     = r1/alpha
    dm1dt   = r0 - r1 - r2
    
    beta1 = beta1m * eval(sets[0]) #* (mue/muemax)
    beta2 = beta2m * ind * eval(sets[1]) #* (mue/muemax)

    de1dt = beta1 - (e1 * mue)
    de2dt = beta2 - (e2 * mue)
        
    return [dm1dt, de1dt, de2dt]


# =============================================================================
# call jacobian and derivatives
# =============================================================================
def jac(z1, par):
    
    m1_r      = z1[-1, 0]
    e1_r      = z1[-1, 1]
    e2_r      = z1[-1, 2]
    
    r0_r      = par[0]
    kcat1_r   = par[1]
    kcat2_r   = par[2]
    Km1_r     = par[3]
    Km2_r     = par[4] 
    beta1m_r  = par[5]
    beta2m_r  = par[6]
    ind_r     = par[7]
    kt1_r     = par[8]
    kt2_r     = par[9]   
    alpha_r   = par[10]  
    sets      = par[11]  #(b1, b2)
    
    m1, e1, e2, r0, kcat1, kcat2, Km1, Km2, beta1m, beta2m, ind, kt1, kt2, alpha  = sp.symbols('m1, e1, e2, r0, kcat1, kcat2, Km1, Km2, beta1m, beta2m, ind, kt1, kt2, alpha')
   
    r1      = kcat1 * e1 * (m1/(m1+Km1))
    r2      = kcat2 * e2 * (m1/(m1+Km2))
    
    mue     = r1/alpha
    dm1dt   = r0 - r1 - r2
    
    beta1 = beta1m * eval(sets[0]) #* (mue/muemax)
    beta2 = beta2m * ind * eval(sets[1]) #* (mue/muemax)

    de1dt = beta1 - (e1 * mue)
    de2dt = beta2 - (e2 * mue)
    
    F = sp.Matrix([dm1dt, de1dt, de2dt])
    
    J = F.jacobian([m1, e1, e2]).subs   ([      (m1, m1_r), (e1, e1_r), (e2, e2_r), (r0, r0_r), (kcat1, kcat1_r), (kcat2, kcat2_r), (Km1, Km1_r), (Km2, Km2_r),
                                                (beta1m, beta1m_r), (beta2m, beta2m_r), (ind, ind_r), (kt1,  kt1_r),(kt2,  kt2_r), (alpha, alpha_r)           ])
    
    DM1 = dm1dt.subs                    ([      (m1, m1_r), (e1, e1_r), (e2, e2_r), (r0, r0_r), (kcat1, kcat1_r), (kcat2, kcat2_r), (Km1, Km1_r), (Km2, Km2_r),
                                                (beta1m, beta1m_r), (beta2m, beta2m_r), (ind, ind_r), (kt1,  kt1_r),(kt2,  kt2_r), (alpha, alpha_r)           ])
    
    DE1 = de1dt.subs                    ([      (m1, m1_r), (e1, e1_r), (e2, e2_r), (r0, r0_r), (kcat1, kcat1_r), (kcat2, kcat2_r), (Km1, Km1_r), (Km2, Km2_r),
                                                (beta1m, beta1m_r), (beta2m, beta2m_r), (ind, ind_r), (kt1,  kt1_r),(kt2,  kt2_r), (alpha, alpha_r)           ])
    
    DE2 = de2dt.subs                    ([      (m1, m1_r), (e1, e1_r), (e2, e2_r), (r0, r0_r), (kcat1, kcat1_r), (kcat2, kcat2_r), (Km1, Km1_r), (Km2, Km2_r),
                                                (beta1m, beta1m_r), (beta2m, beta2m_r), (ind, ind_r), (kt1,  kt1_r),(kt2,  kt2_r), (alpha, alpha_r)           ])

    return np.matrix(J, dtype='float'), [DM1, DE1, DE2]

# =============================================================================
# functions
# =============================================================================
def loguniform(mean = 2, SD = 1.2, size = 1, Max = 20, Min = 0):
    q=-1
    while q<Min or q>Max:
        q = np.random.lognormal(1,1,1)
    return q[0]