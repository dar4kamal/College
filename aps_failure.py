#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:13:29 2018

@author: dar4_kamal
"""
import numpy as np 
import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
#from sklearn.ensemble import weight_boosting

#ww = weight_boosting.AdaBoostClassifier().learning_rate

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv(r"aps_failure_training_set_processed_8bit.csv",sep=",")

X = data.iloc[:,1:] 
Y = data.iloc[:,0] 

Y[Y<0],Y[Y>0] = 0,1

sfk = StratifiedKFold(n_splits=5)
predicted = np.zeros(Y.shape[0])
all_prediction = pd.DataFrame([np.zeros(Y.shape[0]) for i in range(6)])

LogReg = LogisticRegression()
Extr_tree = ExtraTreeClassifier()
D_tree = DecisionTreeClassifier()
Rnd_frst = RandomForestClassifier()
Gboost = GradientBoostingClassifier()
Xgboost = XGBClassifier()#reg_lambda=2

#i=0
for train,test in sfk.split(X,Y):
#    print(i)
    x_train = X.iloc[train,:]
    x_test = X.iloc[test,:]
    y_train = Y.iloc[train]
    y_test = Y.iloc[test]
    
    LogReg.fit(x_train,y_train)
    all_prediction.iloc[0,test] = LogReg.predict(x_test)
#    
    Extr_tree.fit(x_train,y_train)
    all_prediction.iloc[1,test] = Extr_tree.predict(x_test)
#    
    D_tree.fit(x_train,y_train)
    all_prediction.iloc[2,test] = D_tree.predict(x_test)
#    
    Rnd_frst.fit(x_train,y_train)
    all_prediction.iloc[3,test] = Rnd_frst.predict(x_test)
#    
    Gboost.fit(x_train,y_train)
    all_prediction.iloc[4,test] = Gboost.predict(x_test)
    
    Xgboost.fit(x_train,y_train)
    all_prediction.iloc[5,test] = Xgboost.predict(x_test)
#    i += 1

print(accuracy_score(Y,all_prediction.iloc[:,:].values[5])*100)
conf = confusion_matrix(Y,all_prediction.iloc[:,:].values[5])
# for xbgoost accuracy 


#to get all models' accuracy 

#index =0 
#for i in all_prediction.iloc[:,:].values:    
#    print(accuracy_score(Y,i)*100," at ",index)
#    print(confusion_matrix(Y,i))
#    print("---")
#    index += 1