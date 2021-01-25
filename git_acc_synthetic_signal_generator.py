# -*- coding: utf-8 -*-
"""
Function Documentation
======================================
https://matousc89.github.io/signalz/_modules/signalz/generators/ecgsyn.html
"""
#%%
import numpy as np
from signalz.misc import ode45, rem
from signalz import gaussian_white_noise
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from step_count_developed_algorithm import segmented_signal
from step_count_developed_algorithm import filtered_signal

#%%
def template_details(dataset,participant):
    
    # dataset contains the templates associated for each participant - e.g. HW_val_normal[4]
    template = np.reshape(dataset[participant-1],-1)
    template_initial_value = round(template[0],12)
    template_length = len(template)
         
    return template,template_initial_value,template_length
#%%
def original_signal_details(real_dataset,activity_label,type_or_task,participant):

    # information about the original signal that you want to re-generate
    
    # synthetic_signal_validation(val[1],HW_type[4],6,"task")
    signal = segmented_signal(real_dataset,type_or_task,activity_label)
    f_signal = filtered_signal(signal)

    for acc in f_signal[(participant-1):participant]:
        
        real_length = len(acc)
        real_max = round(np.max(acc),3)
        real_min = round(np.min(acc),3)        

    return real_length, real_max, real_min

#%%
def synthetic_real_curve_fit(template,template_length,initial_guess,group,case):

    tfit = np.arange(0,template_length,1)   
    
    # curve fit of template
    k_fit, kcov = curve_fit(fit_func,tfit,template,p0=initial_guess,maxfev=9000)
    
    # create fitting data
    fit = fit_func(tfit,k_fit[0],k_fit[1],k_fit[2],k_fit[3],k_fit[4],k_fit[5],k_fit[6],k_fit[7],k_fit[8],k_fit[9],k_fit[10],k_fit[11])

    print("ai=[",round(k_fit[0],4),",",round(k_fit[1],4),",",round(k_fit[2],4),",",round(k_fit[3],4),"],",
                  "thetai=[",round(k_fit[4],4),",",round(k_fit[5],4),",",round(k_fit[6],4),",",round(k_fit[7],4),"],",
                  "bi=[",round(k_fit[8],4),",",round(k_fit[9],4),",",round(k_fit[10],4),",",round(k_fit[11],4),"])")    

    
    plt.figure(figsize=(5.5,4.5))
    plt.title("A graph to show Gaussian fitting of a particiapnt\nfrom {} group (Case {})".format(group,case),fontsize=15)
    plt.xlabel("Index",fontsize=14)
    plt.ylabel("Gait cycle template\nacceleration (g)",fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.plot(tfit, template, 'ro', label='data')
    plt.plot(tfit, fit, 'b-', label='fit')
    plt.legend(loc='best')
    plt.grid(b=True,color='lightgrey')
    plt.tight_layout()   
    
#%%
def fit_func(tfit, a1, a2, a3, a4, theta1, theta2, theta3, theta4, b1, b2, b3, b4):   
    
    # Function that returns dxdt, dydt, dzdt computed from an ODE for ai, bi and thetai parameters
    def myode(tspan, x_y_z, a1, a2, a3, a4, theta1, theta2, theta3, theta4, b1, b2, b3, b4):
                
        theta = np.arctan2(x_y_z[1], x_y_z[0])
        a0 = 1. - np.sqrt((x_y_z[0]**2) + (x_y_z[1]**2))
        walking_frequency = 1.4
        w0 = 2 * np.pi * walking_frequency

        dtheta1 = rem(theta - theta1, 2. * np.pi) 
        dtheta2 = rem(theta - theta2, 2. * np.pi) 
        dtheta3 = rem(theta - theta3, 2. * np.pi) 
        dtheta4 = rem(theta - theta4, 2. * np.pi)
        
        dxdt = a0*x_y_z[0] - w0*x_y_z[1]        
        dydt = a0*x_y_z[1] + w0*x_y_z[0]   
        dzdt = -((a1 * dtheta1 * np.exp(-0.5*((dtheta1)**2/ (b1**2))))+(a2 * dtheta2 * np.exp(-0.5*((dtheta2)**2/ (b2**2))))+\
             (a3 * dtheta3 * np.exp(-0.5*((dtheta3)**2/ (b3**2))))+(a4 * dtheta4 * np.exp(-0.5*((dtheta4)**2/ (b4**2))))) - x_y_z[2] 

        return np.array([dxdt, dydt, dzdt])
    
    theta_1 = theta1 * (np.pi / 180.)
    theta_2 = theta2 * (np.pi / 180.)
    theta_3 = theta3 * (np.pi / 180.)
    theta_4 = theta4 * (np.pi / 180.)
        
    # healthy
    # participant 1 - [0.334858061525,108]
    # participant 2 - [0.259842518634,110]
    # participant 12 - [0.306580498709,106]
    
    # simulated
    # participant 6 (B) - [0.306150449299,130]
    # participant 7 (C) - [0.130143592852,126]
    # participant 15 (A) - [0.337637264698,182]
    
    # parameters = [template_initial_value,template_length]
    parameters = [0.403129058028,202]
    
    z0 = parameters[0]
    x_y_z = np.array([1,0,z0])
    
    template_len = parameters[1]
    tspan = np.arange(0,template_len*0.01,0.01)
    zsol = ode45(myode, tspan, x_y_z, a1, a2, a3, a4, theta_1, theta_2, theta_3, theta_4, b1, b2, b3, b4)
    
    return zsol[:,2]        
#%%        
def derfunc(t, x_y_z, thetai, ai, bi, walk_frequency):

    # theta 
    theta = np.arctan2(x_y_z[1], x_y_z[0]) #+ np.random.normal(loc=0,scale=round(random.uniform(0,1), 2)) 
    # alpha
    a0 = 1. - np.sqrt((x_y_z[0]**2) + (x_y_z[1]**2))    
    walking_frequency = walk_frequency + random.uniform(-0.1,0.1)
    
    w0 = 2 * np.pi * walking_frequency

    dthetai = rem(theta - (thetai), 2. * np.pi) 

    # x
    dxdt = a0*x_y_z[0] - w0*x_y_z[1]
    # y
    dydt = a0*x_y_z[1] + w0*x_y_z[0]
    
    # added gaussian white noise to the amplitude of z signal 
    # (offset, std)
    
    # healthy (1) - (0,0) 
    # healthy (2) - (0,0)
    # healthy (12) - (0,0)
    
    # simulated (6) (B) - (0,0.35)
    # simulated (7) (C) - (0,0.05)
    # simulated (15) (A) - (0,0.1)
    
    noise_a = gaussian_white_noise(1, offset = 0, std=0.1) 
    ai_noise = ai + noise_a
    
    # z
    dzdt = -np.sum(ai_noise * dthetai * np.exp(-0.5*((dthetai)**2/ bi**2))) - (x_y_z[2])     

    return np.array([dxdt, dydt, dzdt])
#%%   
def acc_syn(z_init, idx_time, min_a, max_a, walk_frequency,ai, thetai, bi):
    
    # amp = amplitude (ai)
    # cen = centre (thetai)
    # wid = width (bi)
    """
    ACC_SYN - realistic walking acceleration generator.

    Kwargs:

    * 'sfacc' : Acceleration sampling frequency (int), in Hz

    * `thetai` : angles of PQRST extrema (1d array of size 5) in degrees

    * `ai` : z-position of PQRST extrema (1d array of size 5)

    * `bi` : Gaussian width of peaks (1d array of size 5)
        
    Returns:

    * `x` : acceleration values in G    
    """
    
    # data cleaning
    thetai = np.array(thetai)
    thetai = thetai * (np.pi / 180.)

    ai = np.array(ai)
    
    bi = np.array(bi)    
 
    # integrate system using fourth order Runge-Kutta
    
    # x_y_z is the initial condition
    x0 = 1
    y0 = 0
    z0 = z_init
    x_y_z = np.array([x0,y0,z0])
    
    # sampling frequency
    sfacc = 100
    initial_time = 0
    dt = 1 / float(sfacc)
    end_time = idx_time*dt # this variable varies according to the length of the real signal
    
    Tspan = np.arange(initial_time, end_time, dt)

    z = ode45(derfunc, Tspan, x_y_z, thetai, ai, bi, walk_frequency)

    min_acc = min_a
    max_acc = max_a
    out = np.interp(z[:,2], (z[:,2].min(), z[:,2].max()), (min_acc, max_acc))
    
    noise = gaussian_white_noise(idx_time, offset = 0, std=0)
    out_and_noise = out + noise
    
    index = 100 * Tspan
    
    # Signal plot
    plt.figure()
    plt.plot(index,z[:,2])#,linewidth=4)
    
#    plt.figure()
#    plt.plot(index,out)#,linewidth=4)

#    plt.plot(index,1z[:,0])

#    print(z[:,2])
    # Trajectory plot
    
    plt.figure()
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = (z[:,2])
    xline = (z[:,0])
    yline = (z[:,1])
    ax.plot3D(xline, yline, zline, 'gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    return z,out_and_noise
