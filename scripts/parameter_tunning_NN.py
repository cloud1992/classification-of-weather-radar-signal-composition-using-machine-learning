#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 20:19:21 2021

@author: Arturo Collado Rosell
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import optuna 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.backend import clear_session
N_cat = 4

dirName = 'training_data/'
try:
    # meta_params = np.load(dirName + 'some_params_to_train.npy')
    # operation_mode = str(meta_params[3])
    training_data = np.load(dirName + 'training_data.npy')
    M = training_data.shape[1]-1
    X = training_data[:,:M].copy()
    y = training_data[:,M:].copy()
    del training_data
except Exception as e:
    print(e)     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 29)
y_train_cat = tf.keras.utils.to_categorical(y_train[:,0], N_cat)
y_test_cat = tf.keras.utils.to_categorical(y_test[:,0], N_cat)

BS = 512
EPOCH = 100
device = '/GPU:0'
def objective(trial):
    clear_session()
    input_layer = Input(shape = (M,1)) 
    x = Conv1D(5,5, activation = 'relu')(input_layer)
    x = Conv1D(5,5, activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(40, activation='relu')(x)
    output_layer = Dense(N_cat, activation='softmax')(x)
    model = Model(inputs = input_layer, outputs = output_layer, name = 'weather_radar_composition_classification_NN' )
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log = True)
    with tf.device(device):
        
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            min_delta=0.005, 
            mode='auto',
            restore_best_weights = True
            )
    

        H = model.fit(X_train, y_train_cat, epochs = EPOCH, batch_size = BS,
              validation_data = (X_test, y_test_cat),
              verbose = 0,
              callbacks=[custom_early_stopping])
        score = model.evaluate(X_test, y_test_cat, verbose=0)
    
    
    return score[1]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10 )
    best_params = study.best_params
    print(best_params)

    
    