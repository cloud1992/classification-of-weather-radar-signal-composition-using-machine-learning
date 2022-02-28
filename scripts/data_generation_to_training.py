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
     
np.random.shuffle(data_PSD)
Batch_size = 512
number_of_batch = data_PSD.shape[0]//Batch_size 
for i in range(number_of_batch):
    np.save(dirName + f"{i}_batch",data_PSD[i*Batch_size : (i+1)*Batch_size   ,:])
np.save(dirName + f"{number_of_batch}_batch",data_PSD[(i+1)*Batch_size:, :])    
     
     
np.save(dirName + 'training_data', data_PSD) 

M = data_PSD.shape[1] - 1
np.save(dirName + 'some_params_to_train',(M, number_of_batch, Batch_size, radar_mode)) 


