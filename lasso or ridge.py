# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:25:38 2020

@author: du
"""

from sklearn import linear_model as lm
import numpy as np

train_data = pd.read_csv('c:/users/du/Desktop/SPRING 2020/Machine learning/projecr/train/train.csv')
train_ans = pd.read_csv('c:/users/du/Desktop/SPRING 2020/Machine learning/projecr/train_answers.csv')

model = lm.lasso(alpha = 0.1) #lm.ridge(alpha = 0.1) 
model.fit(X_train,Y_train)
display(model.intercept_)
display(model.coef_)
y_pred = model.predict(X_test)