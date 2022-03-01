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
from Data_generator_class import DataGenerator

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


dirName_models = 'models/xgboost/'
if not os.path.exists(dirName_models):
     os.mkdir(dirName_models)
     print("Directory " , dirName_models ,  " Created ")
else:    
     print("Directory " , dirName_models ,  " already exists")

params = { 'random_state':29,'n_estimators':1000}
def objective(trial):
    
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
            **params
        )
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)
    
    model.save_model(dirName_models + f"model_{trial.number}_xgb.json")
    preds_test = model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, preds_test)
    
    return accuracy_xgb

#start parameter study

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective, n_trials=100)   
df = study_xgb.trials_dataframe()
df.to_csv(dirName_models + "optuna_study_xgb.csv")

best_parms_xgb = study_xgb.best_params
print('The best parameters are: ')
print(best_parms_xgb) 

