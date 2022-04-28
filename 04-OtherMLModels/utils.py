#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/04/28 20:40:25
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2022 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
'''

# here put the import lib
import numpy as np


def load_data():
    
    # load_data
    data_2duv = np.load('/data1/qzhang/small_dataset_pnas/original/original_dataset.npz')['twoduv']
    labels = np.load('/data1/qzhang/small_dataset_pnas/original/original_dataset.npz')['labels']

    # preprocess labels
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    labels_cat = encoder.fit_transform(labels)


    # split data
    from sklearn.model_selection import train_test_split
    train_x,test_x,train_label,test_label = train_test_split(data_2duv.reshape(-1,161,161,1),labels_cat,
                                                                stratify = labels_cat,
                                                                test_size=0.2,
                                                                random_state=42)
    mean = train_x.mean()
    std = train_x.std()

    train_x -= mean
    train_x /= std
    train_x = train_x.clip(-2,2) 
    
    test_x -= mean
    test_x /= std
    test_x = test_x.clip(-2,2)
    
    return train_x,test_x,train_label,test_label



