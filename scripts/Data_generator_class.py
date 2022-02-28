#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:07:33 2022

@author: Arturo Collado Rosell
"""

import numpy as np
import tensorflow.keras as keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=512, dim=(64), n_channels=1,
                 n_classes= 4, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) 

    def __getitem__(self, index):
        'Generate one batch of data'
        
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim))
        y = np.empty((self.batch_size,1), dtype=int)
        #print(index)
        # Generate data
        aux = np.load('training_data/' + f"{index}_batch" + '.npy')
        X = aux[:, 0:-1]

        # Store class
        y = aux[:, -1]
        y_cat = keras.utils.to_categorical(y, num_classes=self.n_classes)    
        
        return X, y_cat