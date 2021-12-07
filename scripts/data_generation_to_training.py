#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:11:42 2021

@author: Arturo Collado Rosell
"""


import os
import numpy as np
import synthetic_weather_data_IQ


#############################Staggered data####################################
Input_params_stagg = {'M':64,
                'Fc': 5.6e9,
                'Tu':0.25e-3,
                'theta_3dB_acimut':1,
                'radar_mode':'staggered'
                }    
    
data_PSD_stagg, radar_mode = synthetic_weather_data_IQ.synthetic_data_train(**Input_params_stagg) 

############################Uniform data#######################################
# Input_params_uniform = {'M':64,
#                 'Fc': 5.6e9,
#                 'Tu':0.25e-3,
#                 'theta_3dB_acimut':1,
#                 'radar_mode':'uniform',
#                 'L': 10
#                 }    
    
# data_PSD_uniform, radar_mode = synthetic_weather_data_IQ.synthetic_data_train(**Input_params_uniform) 



dirName = 'training_data/'
if not os.path.exists(dirName):
     os.mkdir(dirName)
     print("Directory " , dirName ,  " Created ")
else:    
     print("Directory " , dirName ,  " already exists")
np.save(dirName + 'training_data', data_PSD_stagg) 
# np.save(dirName + 'some_params_to_train',(N_vel, N_s_w, N_csr, radar_mode)) 
