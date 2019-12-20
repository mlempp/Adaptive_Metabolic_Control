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
from datetime import date
from datetime import datetime as timer


from ODEmodel_parameter_sampling_utils import ODEmodel, jac, loguniform


# =============================================================================
# define parameter ranges and run 5000 parameter samplings
# =============================================================================
N = 1000

setss           = [ ['1'    , '(m1)/((m1)+kt1)'    , '(kt1)/((m1)+kt1)'],        #b1
                    ['1'    , '(m1)/((m1)+kt2)'    , '(kt2)/((m1)+kt2)']]        #b2
   

stable_setting = {'m1':[],
                 'e1':[], 
                 'e2':[], 
                 'r1':[], 
                 'r2':[], 
                 'mue':[], 
                 'kcat1':[], 
                 'kcat2':[], 
                 'Km1':[], 
                 'Km2':[], 
                 'ind':[],
                 'kt1':[],
                 'kt2':[],
                 'EV':[],
                 'set':[]}


instable_setting= {'m1':[],
                 'e1':[], 
                 'e2':[], 
                 'r1':[], 
                 'r2':[], 
                 'mue':[], 
                 'kcat1':[], 
                 'kcat2':[], 
                 'Km1':[], 
                 'Km2':[], 
                 'ind':[],
                 'kt1':[],
                 'kt2':[],
                 'EV':[],
                 'set':[]}


z0      = [2.1,     #m1
           0.024,   #e2
           0.0]     #e3


for a in setss[0]:              #iterate through all logics for b1
    for b in setss[1]:          #iterate through all logics for b2
        hsmf=[a,b]
        print(hsmf)
        strain  = 'allLogics'
        
        q = 0    
        while q < N:
            
            kcat1   = np.random.uniform(15720, 16440)
            kcat2   = np.random.uniform(1621, 1789)
            Km1     = np.random.uniform(0.72, 1.06)
            Km2     = np.random.uniform(0.48, 0.60)
            ind     = loguniform()
            kt1     = np.random.uniform(1.5, 2.0)
            kt2     = np.random.uniform(1.5, 2.0)  
        
         
            par     = [87.5,                 #0 r0
                       kcat1,               #1 kcat1
                       kcat2,               #2 kcat2
                       Km1,                 #3 Km1
                       Km2,                 #4 Km2
                       0.00024,             #5 beta1
                       0.00068,             #6 beta1
                       ind,                 #7 ind
                       kt1,                 #8 kt1
                       kt2,                 #9 kt2
                       8750,               #10 alpha
                       hsmf]                #11 set   (b1, b2)
        
            x       = 10000
            t       = np.linspace(0, x, num=(x*10)+1)
                
            z1      = odeint(ODEmodel, z0, t, args=(par, )  )
            J,SS    = jac(z1, par)
            
            EV      = np.max(np.real(linalg.eig(J)[0]))
        
            # check if steady state is reached
            if np.sum(np.absolute(SS)) < 1e-08  and EV < -1e-08 and all(z1[:,0]>=0) and all(z1[:,1]>=0) and all(z1[:,2]>=0):
                
                if q % 20 ==0:
                    print(q)
                q += 1
                m1 = z1[:,0]
                e1 = z1[:,1]
                e2 = z1[:,2]
                  
                r1          = par[1] * e1 * (m1/(m1+par[3]))
                r2          = par[2] * e2 * (m1/(m1+par[4]))
                mue         = r1/par[10]
                
                stable_setting['m1'].append(m1[-1])
                stable_setting['e1'].append(e1[-1])
                stable_setting['e2'].append(e2[-1])
                stable_setting['r1'].append(r1[-1])
                stable_setting['r2'].append(r2[-1])
                stable_setting['mue'].append(mue[-1])
                stable_setting['kcat1'].append(par[1])
                stable_setting['kcat2'].append(par[2])
                stable_setting['Km1'].append(par[3])
                stable_setting['Km2'].append(par[4])
                stable_setting['ind'].append(par[7])
                stable_setting['kt1'].append(par[8])
                stable_setting['kt2'].append(par[9])
                stable_setting['EV'].append(EV)
                stable_setting['set'].append(hsmf)
                
            else:
                
                m1 = z1[:,0]
                e1 = z1[:,1]
                e2 = z1[:,2]
                  
                r1          = par[1] * e1 * (m1/(m1+par[3]))
                r2          = par[2] * e2 * (m1/(m1+par[4]))
                mue         = r1/par[10]
                
                instable_setting['m1'].append(m1[-1])
                instable_setting['e1'].append(e1[-1])
                instable_setting['e2'].append(e2[-1])
                instable_setting['r1'].append(r1[-1])
                instable_setting['r2'].append(r2[-1])
                instable_setting['mue'].append(mue[-1])
                instable_setting['kcat1'].append(par[1])
                instable_setting['kcat2'].append(par[2])
                instable_setting['Km1'].append(par[3])
                instable_setting['Km2'].append(par[4])
                instable_setting['ind'].append(par[7])
                instable_setting['kt1'].append(par[8])
                instable_setting['kt2'].append(par[9])
                instable_setting['EV'].append(EV)
                instable_setting['set'].append(hsmf)
                

# =============================================================================
# save           
# =============================================================================
today   = date.today()
d3      = today.strftime("%y%m%d_")
d2 = timer.now().strftime('%H%M%S')
path    = 'results/'


saving = path+d3+d2+'_parameters_'+strain+'_'+str(q)

np.save(saving, [stable_setting, instable_setting])



           
                            