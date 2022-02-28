#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:09:11 2021

@author: Arturo Collado Rosell
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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

device = '/CPU:0'
# device = '/GPU:0'

#####NN model##################
input_layer = Input(shape = (M,1)) 
x = Conv1D(15,10, activation = 'relu')(input_layer)
x = Conv1D(10,10, activation = 'relu')(x)
x = Flatten()(x)
x = Dense(40, activation='relu')(x)
x = Dense(50, activation='relu')(x)
output_layer = Dense(N_cat, activation='softmax')(x)
model = Model(inputs = input_layer, outputs = output_layer, name = 'weather_radar_composition_classification_NN' )

model.summary()

EPOCHS = 100
BS = 512
lr = 0.00239

plot_dir = 'plot_training/'
if not os.path.exists(plot_dir):
     os.mkdir(plot_dir)
     print("Directory " , plot_dir ,  " Created ")
else:    
     print("Directory " , plot_dir ,  " already exists")

directory_to_save_model = plot_dir  +device[1:4]  + '_' + str(EPOCHS) + '_' + str(BS) 

with tf.device(device):
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            min_delta=0.005, 
            mode='auto',
            restore_best_weights = True
            )
    
    H = model.fit(X_train, y_train_cat, epochs = EPOCHS, batch_size = BS,
          validation_data = (X_test, y_test_cat),
          verbose = 1,
          callbacks=[custom_early_stopping])
    model.save(directory_to_save_model + '_M_' + str(M)+'.h5')
    
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis = 1)    
    confusion = confusion_matrix(y_test, y_pred, normalize = 'true')
    print('Confusion Matrix\n')
    print(confusion)
    
####Plots############################################################
loss_training = H.history['loss']
loss_validation = H.history['val_loss']
accuracy_training = H.history['accuracy']
accuracy_validation = H.history['val_accuracy']

#Loos function
figure = plt.figure()
plt.plot(np.arange(0,len(loss_training)) ,loss_training, label = 'Training')
plt.plot(np.arange(0,len(loss_training)) ,loss_validation, label = 'Validation')
plt.legend()
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(plot_dir + 'Loos.png')
figure.show()

#Accuracy function
figure = plt.figure()
plt.plot(np.arange(0,len(loss_training)) ,accuracy_training, label = 'Training')
plt.plot(np.arange(0,len(loss_training)) ,accuracy_validation, label = 'Validation')
plt.legend()
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(plot_dir + 'Accuracy.png')
figure.show()



