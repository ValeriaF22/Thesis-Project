# -*- coding: utf-8 -*-
"""
STEP COUNT ALGORITHM

Template matching using Dynamic Time Warping 
"""
#%%
import numpy as np
from scipy.signal import find_peaks
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw
from subset_data_by_activity_number import subset_data_by_activity_number
from scipy import signal
import pandas as pd
#%% Autocorrelation
def autocorr(signal):
    
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    len_signal = len(signal)
    
    # normalisation formula 
    # length is used to stretch the signal in y axis to become smaller
    norm_1 = (signal - mean_signal)/(std_signal*len_signal)
    norm_2 = (signal - mean_signal)/std_signal 
    # result is between -1 and 1
    result = np.correlate(norm_1, norm_2, mode='full')
   
    return result[result.size//2:]#/signal.size
#%% Unbiased autocorrelation
def autocorr_unbiased(signal):
    
    vec_unbi = np.array([signal.size - abs(i) for i in range(0, signal.size)])
    
    return signal.size*autocorr(signal)/vec_unbi
    
#%% Selection of specific activities
def segmented_signal(segmented_dataset,type_or_task,activity_label): 

    if type_or_task == "type":
        subset_data = subset_data_by_activity_number(segmented_dataset,"type",activity_label)
    elif type_or_task == "task":
        subset_data = subset_data_by_activity_number(segmented_dataset,"task",activity_label)

    return subset_data
#%% Filtering acceleration signal
def filtered_signal(subset_data):
    
    # Filter the acceleration signal using Butterworth filter
    w = 2 / (100 / 2) # Normalize the frequency, fc is cut-off frequency
    
    b_butter, a_butter = signal.butter(4, w, 'low') # order = 2 for healthy     
    
    f_acc = []
    for idx,acc in enumerate(subset_data):
        
        # filtered acceleration
        f_acc.append(signal.filtfilt(b_butter, a_butter, acc.acc_x))
        
    return f_acc

#%%
def template_length(segmented_signal,group,activity_label,location): 

    template_length = []   
    for idx,acc in enumerate(segmented_signal):

        # this is for only a few special cases 
        acc_range = np.max(acc)-np.min(acc)
        if acc_range > 1: # slow, normal, fast = 1
            updated_acc = np.where(acc<-0.5,np.mean(acc),acc) #healthy & simulated (slow, normal, fast) = -0.15 
        else:
            updated_acc = acc

        # unbiased autocorrelation of filtered acceleration
        a_acc = autocorr_unbiased(updated_acc)
        
        # peaks from autocorrelation signal
        peaks_i, _ = find_peaks(a_acc)
        # troughs from autocorrelation signal
        troughs_i, _ = find_peaks(-a_acc)

        distance_p = peaks_i[1]-peaks_i[0]          
        distance_t = troughs_i[1]-troughs_i[0]
        
        # constants used to calculate distance threshold for each activity and each group
        if group == "h":
            if (activity_label == 5):
                c = 0.65
            else:
                c = 0.1
                                
            peak_distance = int(distance_p*c) 
            trough_distance = int(distance_t*c)
            
        elif group == "s":
            if (activity_label == 3):
                c = 1.2
            elif (activity_label == 5): 
                c = 0.95
            elif (activity_label == 6):
                c = 0.9
            elif (activity_label == 8) or (activity_label == 9):
                c = 1.3 # for up c = 1.2  #0.95
            elif (activity_label == 7):
                c = 0.75
                
            peak_distance = int(distance_p*c) 
            trough_distance = int(distance_t*c) 
        
        # find peaks and troughs using the distance threshold
        peaks_acc, _ = find_peaks(acc,distance=peak_distance)   
        troughs_acc, _ = find_peaks(-acc,distance=trough_distance) 
        
        peaks_acc_diff = np.diff(peaks_acc)
        troughs_acc_diff = np.diff(troughs_acc)

        template_length.append(int(np.mean([np.mean(peaks_acc_diff),np.mean(troughs_acc_diff)])))
        

    return template_length

#%%
def template_signal(signals):

    template_signals = []
    for participant in signals:
              
        template_signals.append(dtw_barycenter_averaging(participant, max_iter=0))                 

    return template_signals    

#%%
def slidingWindow(sequence,window_size,step=1):

    num_windows = int(((len(sequence)-window_size)/step)+1)

    windows = []
    for i in range(0,num_windows*step,step):
        
        windows.append(sequence[i:i+window_size])

    return windows

#%%
def acc_windows(segmented_signal,template_length):
   
    windows = []
    for idx, acc in enumerate(segmented_signal):
        
        windows.append(slidingWindow(acc,template_length[idx],template_length[idx])) 
        
    return windows

#%%
def dtw_similarity(template_signal,acc_signal_windows):
    
    dist_similarity = []
    for i, v in enumerate(acc_signal_windows):
                
        dist_similarity.append([dtw(w, template_signal[i]) for w in v])

    return dist_similarity
#%%
def tmp_steps(similarity_measures,true_steps,group,location,activity_label):

    steps = []

    for i,v in enumerate(similarity_measures):
            
        max_value = np.max(v) 
        min_value = np.min(v)
        mid_range_value = (max_value+min_value)/2
        
        # constants used to calculate distance threshold for each activity and each group
        if group == "h":
            c = 0.9
            
        elif group == "s":
            if (activity_label == 3):
                c = 0.6
            elif (activity_label == 5): 
                c = 0.9
            elif (activity_label == 6) or (activity_label == 7):
                c = 0.4
            elif (activity_label == 8) or (activity_label == 9):
                c = 0
                
        threshold = mid_range_value+(mid_range_value*c)

        high_threshold = threshold
        steps.append([value for value in v if value <= high_threshold])         

    num_steps = [np.size(step) for step in steps]
    
    return np.array(num_steps),similarity_measures
#%%
def developed_algorithm_steps(dataset,activity_label,group,location,type_or_task,true_steps):
    
    # acceleration signal of interest
    signal = segmented_signal(dataset,type_or_task,activity_label)
    # filtered acceleration signal
    f_signal = filtered_signal(signal)
    # length of a single template
    tl_signal = template_length(f_signal,group,activity_label,location)
    # acceleration signal segmented into several windows with the same length as their associate template
    windows = acc_windows(f_signal,tl_signal)     
    # template for a single step
    step_template = template_signal(windows)
    # checking similarity between the step template and each window    
    dtw = dtw_similarity(step_template,windows)
    # number of steps calculated 
    steps = tmp_steps(dtw,true_steps,group,location,activity_label)
    
    # visualise in console the true and predicted number of steps, as well as their difference
    # initialize list of lists 
    size_participant = np.size(steps[0])
    data = [list(true_steps),steps[0],list(true_steps-steps[0])]
    
    # Create the pandas DataFrame 
    df = pd.DataFrame(data, index = ['true', 'sim', 'diff'])
    pd.set_option('display.max_columns', size_participant)
    print("----")
    print(df)
    
    doubled_tl_signal = [num*2 for num in tl_signal]
    doubled_windows = acc_windows(f_signal,doubled_tl_signal) 
    doubled_template = template_signal(doubled_windows)

    return steps,windows,dtw,step_template,doubled_template,doubled_tl_signal

