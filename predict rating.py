#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:53:39 2020

@author: regitafach
"""

import streamlit as st

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
plt.style.use('fivethirtyeight')

#import data
data = pd.read_csv('rating_top.csv',index_col=[0], parse_dates=[0])
data.head()

#Numerical-Categorical Split
x_num = data.drop(['GENRE','MOVIE'], axis=1)
label = ['GENRE','MOVIE']
x_cat = data[label]

#Missing value checking
x_num.isnull().any()
x_cat.isnull().any()

#Create new feature
x_num['date'] = x_num.index
x_num['DAYOFWEEK'] = x_num['date'].dt.dayofweek
x_num['QUARTER'] = x_num['date'].dt.quarter
x_num['MONTH'] = x_num['date'].dt.month
x_num['YEAR'] = x_num['date'].dt.year
x_num['DAYOFYEAR'] = x_num['date'].dt.dayofyear
x_num['DAYOFMONTH'] = x_num['date'].dt.day
x_num['WEEKOFYEAR'] = x_num['date'].dt.weekofyear

#categorical dummy
x_cat=pd.get_dummies(x_cat[label])

#Combine Categorical and Numerical Data
x_new = pd.concat([x_num, x_cat], axis=1)
x_new.head()

#Cek sebaran data
cek_sebaran = data.drop(['GENRE','MOVIE'], axis=1)
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = cek_sebaran.plot(style='.', figsize=(15,5), color=color_pal[0], title='Rating')

#Training testing split
split_date = '2019-09-01'
data_train = x_new.loc[x_new.index < split_date].copy()
data_test = x_new.loc[x_new.index >= split_date].copy()

x_train = data_train.drop(["Rating", "date"], axis = 1)
x_test = data_test.drop(["Rating", "date"], axis = 1)
y_train = data_train["Rating"]
y_test = data_test["Rating"]

#Create Random Forest Model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import check_array

def bestparam_randCV(model,hyperparam,x_train, y_train, n_iter=1000):
    
    hyperparam = hyperparam
    randomizedCV = RandomizedSearchCV(model, param_distributions = hyperparam, cv = 10,
                                          n_iter = n_iter, scoring = 'neg_mean_squared_error', n_jobs=-1, 
                                          random_state = 42, verbose = True)
    randomizedCV.fit(x_train, y_train)
    
    #print (randomizedCV.cv_results_)
    print ('Best MSE', randomizedCV.score(x_train, y_train))
    print ('Best Param', randomizedCV.best_params_)
    return randomizedCV

reg         = RandomForestRegressor(n_estimators=1000)             

hyperparam = {'bootstrap': [True, False],
             'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
             'max_features': ['auto', 'sqrt'],
             'min_samples_leaf': [1, 2, 4],
             'min_samples_split': [2, 5, 10],
             'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}
n_iter     = 10 #20
best_rf1   = bestparam_randCV(reg, hyperparam, x_train, y_train, n_iter)

rf1 = RandomForestRegressor(bootstrap     = best_rf1.best_params_.get('bootstrap'),
                        max_features      = best_rf1.best_params_.get('max_features'),
                        max_depth         = best_rf1.best_params_.get('max_depth'),
                        n_estimators      = best_rf1.best_params_.get('n_estimators'),
                        min_samples_split = best_rf1.best_params_.get('min_samples_split'),
                        min_samples_leaf  = best_rf1.best_params_.get('min_samples_leaf'))

result_rf1 = rf1.fit(x_train, y_train)

#Feature Importances
## Jika tidak jelas grafiknya ambil file .png saja yang di samping nama filenya "filename.png"
importances = result_rf1.feature_importances_
indices = np.argsort(importances)
features=x_train.columns
plt.figure(figsize=(20,50))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.savefig('filename.png', dpi=300)

#Forecast on Test Set
data_test['RATING_Prediction'] = result_rf1.predict(x_test)
data_all = pd.concat([data_test, data_train], sort=False)

_ = data_all[['Rating','RATING_Prediction']].plot(figsize=(15, 5))

#Error Metrics on Test Set
#RMSE
sqrt(mean_squared_error(y_true=data_test['Rating'],
                   y_pred=data_test['RATING_Prediction']))
#MAE
mean_absolute_error(y_true=data_test['Rating'],
                   y_pred=data_test['RATING_Prediction'])
#MAPE
mean_absolute_percentage_error(y_true=data_test['Rating'],
                   y_pred=data_test['RATING_Prediction'])


