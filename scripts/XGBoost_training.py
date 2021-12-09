#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:51:36 2021

@author: Arturo Collado Rosell
Training using XGBoost
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import optuna 

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



def objective(trial, X_train, X_test, y_train, y_test, **parms):
    
    learning_rate = trial.suggest_float("learning_rate", 0.05e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)
    
    model = xgb.XGBClassifier(
            tree_method="gpu_hist",
            gpu_id=0,
            predictor="gpu_predictor",
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            max_depth=max_depth,
            use_label_encoder=False,
            **parms
        )
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)
    preds_test = model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, preds_test)
    return accuracy_xgb

#start parameter study
params = { 'random_state':29,'n_estimators':1000}
study1 = optuna.create_study(direction="maximize")
study1.optimize(lambda trial:objective(trial,X_train, X_test, y_train, y_test, **params), n_trials=10)   

best_parms1 = study1.best_params
print('The best parameters are: ')
print(best_parms1) 

##############XGboot model################################################
# data_dmatrix = xgb.DMatrix(data = X_train, label = y_train)
xg_reg = xgb.XGBClassifier(**best_parms1,**params, tree_method = 'gpu_hist', predictor='gpu_predictor', use_label_encoder=False)  
  
# xg_reg = xgb.XGBClassifier(tree_method = 'gpu_hist', predictor='gpu_predictor', objective = 'multi:softmax', num_class = 4, colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 100, use_label_encoder=False  )   

xg_reg.fit(X_train, y_train)

y_pred_xgb = xg_reg.predict(X_test)
predictions = [round(value) for value in y_pred_xgb]
accuracy_xgb = accuracy_score(y_test, predictions)
print(f'The accuracy using XGBoost is {accuracy_xgb}')