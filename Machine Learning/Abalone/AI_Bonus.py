#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 08:58:39 2018

@author: dar4_kamal
"""

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor


file = open(r"/home/dar4_kamal/Downloads/Machine Learning/Abalone/abalone.domain","r").read().splitlines()
col_names = [line.split(":")[0] for line in file]
df = pd.read_csv(r"/home/dar4_kamal/Downloads/Machine Learning/Abalone/abalone.data",sep = ",",names=col_names)

def catgorize(x):
    typ = x.dtype
    le = LabelEncoder()
    if(typ=="object"):
        le.fit(x)
        x = le.transform(x)
    x = (x-x.mean())/(x.max()-x.min())
    return x
    
#df_ct = df.apply(catgorize,axis=0)
#df_fs = df_ct.apply(Feature_scaling,axis=0)
X = df.iloc[:,0:df.shape[1]-1]
X = X.apply(catgorize,axis=0)
Y = df.rings

rnd = []
RMSE = []

LinregL = []
extr_treeL = []
D_treeL = []
Rnd_forstL = []
GredBoostL = []
XGBL = []

indexq = [n/100 for n in range(40,71)]
pdf = pd.DataFrame(index=indexq,columns=["LinregL","D_treeL","extr_treeL","Rnd_forstL","GredBoostL","XGBL"])

start = time.time()
Linreg =LinearRegression()
extr_tree =ExtraTreeRegressor()
D_tree =DecisionTreeRegressor()
Rnd_forst =RandomForestRegressor()
GredBoost =GradientBoostingRegressor(max_depth=3)
XGB =XGBRegressor() 

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size = .333,random_state=42)

Linreg.fit(x_train,y_train)
result_lin = Linreg.predict(x_test)
LinregL.append(np.sqrt(mean_squared_error(y_test,result_lin)))


extr_tree.fit(x_train,y_train)
result_extr_tree = extr_tree.predict(x_test)
extr_treeL.append(np.sqrt(mean_squared_error(y_test,result_extr_tree)))


D_tree.fit(x_train,y_train)
result_D_tree = D_tree.predict(x_test)
D_treeL.append(np.sqrt(mean_squared_error(y_test,result_D_tree)))


Rnd_forst.fit(x_train,y_train)
result_Rnd_forst = Rnd_forst.predict(x_test)
Rnd_forstL.append(np.sqrt(mean_squared_error(y_test,result_Rnd_forst)))


GredBoost.fit(x_train,y_train)
result_GredBoost = GredBoost.predict(x_test)
GredBoostL.append(np.sqrt(mean_squared_error(y_test,result_GredBoost)))

XGB.fit(x_train,y_train)
result_XGB = XGB.predict(x_test)
XGBL.append(np.sqrt(mean_squared_error(y_test,result_XGB)))

end = time.time()

print("Linear_regression err :            ",min(LinregL))
print("Extra_tree_regression err :        ",min(extr_treeL))
print("Decision_tree_regression err :     ",min(D_treeL))
print("Random_Forest_regression err :     ",min(Rnd_forstL))
print("Gradient_Boosting_Regression err : ",min(GredBoostL))
print("XGBoost_regression err :           ",min(XGBL))

print("Duration : ",end-start)
