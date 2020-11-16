# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:45:25 2020

@author: mn12vf
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# given data we want to fit
tspan = [0, 0.1, 0.2, 0.4, 0.8, 1]
Ca_data = [2.0081,  1.5512,  1.1903,  0.7160,  0.2562,  0.1495]

def fitfunc(t, k):
    'Function that returns Ca computed from an ODE for a k'
    def myode(Ca, t):
        return -k * (Ca**3)

    Ca0 = Ca_data[0]
    Casol = odeint(myode, Ca0, t)
    return Casol[:,0]

k_fit, kcov = curve_fit(fitfunc, tspan, Ca_data, p0=1.3)
print(k_fit)

tfit = np.linspace(0,1);
fit = fitfunc(tfit, k_fit)

plt.figure()
plt.plot(tspan, Ca_data, 'ro', label='data')
plt.plot(tfit, fit, 'b-', label='fit')
plt.legend(loc='best')


#%%

tspan = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Ca_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def fitfunc(t, k):
    'Function that returns Ca computed from an ODE for a k'
    def myode(Ca, t):
        
        dydt = k
        
        return dydt

    Ca0 = Ca_data[0]
    Casol = odeint(myode, Ca0, t)
    return Casol[:,0]

k_fit, kcov = curve_fit(fitfunc, tspan, Ca_data, p0=0)
print(k_fit)

tfit = np.linspace(0,10);
fit = fitfunc(tfit, k_fit)

plt.figure()
plt.plot(tspan, Ca_data, 'ro', label='data')
plt.plot(tfit, fit, 'b-', label='fit')
plt.legend(loc='best')

#%%

tspan = [1, 0, -1, 0]#, 4, 5, 6, 7, 8, 9, 10]
Ca_data = [0, 1, 0, -1]#, 4, 5, 6, 7, 8, 9, 10]

def fitfunc(t, k):
    'Function that returns Ca computed from an ODE for a k'
    def myode(Ca, t):
        
        dydt = -Ca/16-Ca**2
        
        return dydt

    Ca0 = Ca_data[0]
    Casol = odeint(myode, Ca0, t)
    return Casol[:,0]

k_fit, kcov = curve_fit(fitfunc, tspan, Ca_data, p0=0)
print(k_fit)

tfit = np.linspace(0,10);
fit = fitfunc(tfit, k_fit)

plt.figure()
plt.plot(tspan, Ca_data, 'ro', label='data')
plt.plot(tfit, fit, 'b-', label='fit')
plt.legend(loc='best')

#%%
new_template = []
template = HW_val_normal[4][0]

for i in range(0,(len(template)-1)):
    tmp_template = np.linspace(template[i],template[i+1],10)
    for j in range(0,len(tmp_template)):
        
        new_template.append(tmp_template[j])

flat_new_template = [k for q in new_template for k in q]
        
tspan = np.linspace(0,107,214)

k_fit, kcov = curve_fit(fit_func,tspan,flat_new_template,p0=[-1,1,-1,1,20,45,70,100,0.01,0.01,0.01,0.01],maxfev=9000)

tfit = np.linspace(0,107,10)
fit = fit_func(tfit,k_fit[0],k_fit[1],k_fit[2],k_fit[3],k_fit[4],k_fit[5],k_fit[6],k_fit[7],k_fit[8],k_fit[9],k_fit[10],k_fit[11])

plt.figure()
#plt.plot(tspan, flat_new_template, 'ro', label='data')
plt.plot(HW_val_normal[4][0], 'r', label='data')
plt.plot(tfit, fit, 'b-', label='fit')
plt.legend(loc='best')
#%%
# EXAMPLE
tspan = np.linspace(0,107,856)
k_fit, kcov = curve_fit(fit_func,tspan,flat_new_template,p0=[-10,10,-10,10,20,45,70,100,1,1,1,1],maxfev=9000)

tfit = np.arange(0,107,0.01)
fit = fit_func(tfit,k_fit[0],k_fit[1],k_fit[2],k_fit[3],k_fit[4],k_fit[5],k_fit[6],k_fit[7],k_fit[8],k_fit[9],k_fit[10],k_fit[11])

"""
plt.plot(xdata, func_4(xdata, *popt), 'r-',label="fit")# a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)) 
"""

plt.figure()
plt.plot(tspan/100, flat_new_template, 'ro', label='data')
plt.plot(tspan, flat_new_template, 'ro', label='data')
plt.plot(tfit, fit, 'b-', label='fit')
plt.legend(loc='best')

HW_acc_syn_acceleration_signal_output = acc_syn(z_init=-0.15,idx_time=560,min_a=-0.292,max_a=0.643,a=0.00025,b=20, th=0,f=0.5,walk_frequency=1.4,ai=[-k_fit[0],-k_fit[1],-k_fit[2],-k_fit[3] ], thetai=[ k_fit[4],k_fit[5],k_fit[6],k_fit[7] ], bi=[ k_fit[8],k_fit[9],k_fit[10],k_fit[11] ])