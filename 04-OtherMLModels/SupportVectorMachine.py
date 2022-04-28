#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   SupportVectorMachine.py
@Time    :   2021/04/28 20:29:39
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2021 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
'''

# here put the import lib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from utils import load_data
from joblib import dump

if __name__ == '__main__':
    
    model_name = 'svm-rbf'
    train_x,test_x,train_label,test_label = load_data()

    model =  SVC(gamma='auto',kernel='rbf',random_state=42)

    model.fit(train_x.reshape(-1,161*161),train_label)
    
    dump(model,'%s.joblib'%model_name)
    
    y_pre = model.predict(test_x.reshape(-1,161*161))
    confusion_matrix = confusion_matrix(test_label,y_pre)
    print(confusion_matrix)
    
    
    # rbg
    model_name = 'svm-rbg'

    model =  SVC(gamma='auto',kernel='rbg',random_state=42)

    model.fit(train_x.reshape(-1,161*161),train_label)
    
    dump(model,'%s.joblib'%model_name)
    
    y_pre = model.predict(test_x.reshape(-1,161*161))
    confusion_matrix = confusion_matrix(test_label,y_pre)
    print(confusion_matrix)
    
    
    # linear
    model_name = 'svm-linear'

    model =  SVC(gamma='auto',kernel='linear',random_state=42)

    model.fit(train_x.reshape(-1,161*161),train_label)
    
    dump(model,'%s.joblib'%model_name)
    
    y_pre = model.predict(test_x.reshape(-1,161*161))
    confusion_matrix = confusion_matrix(test_label,y_pre)
    print(confusion_matrix)
    
    
    # poly
    model_name = 'svm-linear'

    model =  SVC(gamma='auto',kernel='poly',random_state=42)

    model.fit(train_x.reshape(-1,161*161),train_label)
    
    dump(model,'%s.joblib'%model_name)
    
    y_pre = model.predict(test_x.reshape(-1,161*161))
    confusion_matrix = confusion_matrix(test_label,y_pre)
    print(confusion_matrix)
    


