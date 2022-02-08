#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:00:59 2022

@author: acr
"""

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
import matplotlib.pyplot as plt
import RadarNet 

import numpy as np
import tensorflow as tf
import synthetic_weather_data_IQ

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

np.random.seed(2022) # seed for reproducibility 

######Simulation Parameers#####################################################
Tu = 0.25e-3
PRF = 1/Tu
M = 64
Fc = 5.6e9
c = 3.0e8
theta_3dB_acimut = 1
wavelenght = c/Fc
va = wavelenght*PRF/4
Sp = 1
vm = 0.2*va
spectral_w = 0.1*va
# csr = np.linspace(0, 50, 25)
# Sc = Sp * 10**(csr/10)

radar_mode = 'uniform'
snr = np.linspace(0, 30, 15)
power_noise = Sp / (10**(snr/10))

I = 1
#         clutter_power: power of clutter [times]
#         clutter_s_w : clutter spectrum width [m/s] 
#         phenom_power: power of phenom [times] 
#         phenom_vm: phenom mean Doppler velocity [m/s]
#         phenom_s_w: phenom spectrum width [m/s]
#         noise_power: noise power [times] 
#         M: number of samples in a CPI, the sequence length  
#         wavelength: radar wavelength [m] 
#         PRF: Pulse Repetition Frequecy [Hz]
#         radar_mode: Can be "uniform" or "staggered" 
#         int_stagg: list with staggered sequence, by default it is [2,3] when radar_mode == "staggered" 
#         samples_factor: it needed for windows effects, by default it is set to 10
#         num_realizations: number of realization 

parameters_without_clutter = {    
                               'phenom_power':Sp,
                               'phenom_vm': vm,
                               'phenom_s_w': spectral_w,                                        
                               'PRF':PRF,
                               'radar_mode':radar_mode,
                               'M':M,
                               'wavelenght':wavelenght,
                               'num_realizations':I
                               }   
#window
num_samples_uniform = M  
window = np.kaiser(num_samples_uniform, 8)

# 'noise_power': power_noise, 
###########Data generation#####################################
L = 1000 # Monte Carlo realization number
N_snr = len(snr)
complex_IQ_data = np.zeros((N_snr*L, M), dtype = complex)
data_PSD = np.zeros((N_snr*L, num_samples_uniform))
for i in synthetic_weather_data_IQ.progressbar(range(N_snr), 'Computing:') :
 
    parameters_without_clutter['noise_power'] = power_noise[i]
    for ind_l in range(L):
        z_IQ, _ = synthetic_weather_data_IQ.synthetic_IQ_data(**parameters_without_clutter)
        complex_IQ_data[i*L + ind_l,:] = z_IQ        
        
        data_w = z_IQ * window
                        
        ##PSD estimation
        psd = synthetic_weather_data_IQ.PSD_estimation(data_w, w = window, PRF = 1/Tu)
        psd = psd /np.max(psd)
        data_PSD[i*L + ind_l,:] = 10*np.log10(psd[0,:])
                
###########################Predictions########################################## 
#predictions using the NN

model = tf.keras.models.load_model('../plot_training/'+'GPU_100_512_M_64.h5')
class_predicted = model.predict(data_PSD)  
class_predicted = np.argmax(class_predicted, axis = 1)    