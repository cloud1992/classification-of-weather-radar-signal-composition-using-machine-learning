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
radar_mode = "uniform"

Input_params_stagg = {'M':64,
                'Fc': 5.6e9,
                'Tu':0.25e-3,
                'theta_3dB_acimut':1,
                'radar_mode':radar_mode
                }    
    
data_PSD = synthetic_weather_data_IQ.synthetic_data_train(**Input_params_stagg) 



dirName = 'training_data/'
if not os.path.exists(dirName):
     os.mkdir(dirName)
     print("Directory " , dirName ,  " Created ")
else:    
     print("Directory " , dirName ,  " already exists")
np.save(dirName + 'training_data', data_PSD) 

