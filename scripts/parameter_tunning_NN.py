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

import optuna 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.backend import clear_session
from Data_generator_class import DataGenerator
N_cat = 4

dirName = 'training_data/'
try:
    meta_params = np.load(dirName + 'some_params_to_train.npy')

    M = int(meta_params[0])
    number_of_batch = int(meta_params[1])
    batch_size = int(meta_params[2])
    radar_mode = meta_params[3]
except Exception as e:
    print(e)     

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 29)
#y_train_cat = tf.keras.utils.to_categorical(y_train[:,0], N_cat)
#y_test_cat = tf.keras.utils.to_categorical(y_test[:,0], N_cat)


EPOCHS = 120
device = '/GPU:0'


dirName_models = 'models/NN/'
if not os.path.exists(dirName_models):
     os.mkdir(dirName_models)
     print("Directory " , dirName_models ,  " Created ")
else:    
     print("Directory " , dirName_models ,  " already exists")

def create_model(trial):
    n_conv_layers = trial.suggest_int("n_layers_conv",1,4)
    n_dens_layers = trial.suggest_int("n_layers_dens",1,4)

    input_layer = Input(shape = (M,1)) #input layer
    x = input_layer
    # x = Conv1D(10,10, activation = 'relu')(x)
    # x = Flatten()(x)
    # x = Dense(40, activation = 'relu')(x)
    # output_layer = Dense(N_cat, activation= 'softmax')(x)
    for i in range(n_conv_layers):
        x = Conv1D(filters=trial.suggest_categorical(f'filters_{i}', [2, 5, 10, 15]), kernel_size = trial.suggest_categorical(f"kernel_size_{i}", [3, 6, 10, 12]), activation = 'relu')(x)

    x = Flatten()(x)
    
    for i in range(n_dens_layers):
        x = Dense(units = trial.suggest_categorical(f"units_{i}", [30,40, 50, 60]), activation='relu')(x)

    output_layer = Dense(N_cat, activation='softmax')(x)
    model = Model(inputs = input_layer  , outputs = output_layer  , name = 'Radar_signal_classificator')
    return model

training_IDs = [i for i in range(int(number_of_batch*0.8))]
validation_IDs = [j for j in range(int(number_of_batch*0.8),number_of_batch)]

#Generators
params = {"dim": M,
          "batch_size": batch_size,
          "n_classes": N_cat
          }
training_generator = DataGenerator(training_IDs, **params)
validation_generator = DataGenerator(validation_IDs, **params)

def objective(trial):
    clear_session()
    
    model = create_model(trial)
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
    

        H = model.fit(training_generator,
                      validation_data = validation_generator,
                      epochs = EPOCHS,
                      use_multiprocessing=True,
                      workers = -1,
                      verbose = 1,
                      callbacks=[custom_early_stopping])
        model.save(dirName_models + f'_{trial.number}.h5' )
        H_df = pd.DataFrame(H.history)
        H_df.to_csv(dirName_models + f'_{trial.number}' + '.csv')
        score = model.evaluate(validation_generator, verbose=0)
    
    
    return score[1]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100 )
    df = study.trials_dataframe()
    df.to_csv(dirName_models + "optuna_study.csv")
    best_params = study.best_params
    print(best_params)

    
    