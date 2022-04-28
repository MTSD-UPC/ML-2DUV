#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   01-SplitDataset.py
@Time    :   2022/04/27 16:08:26
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2022 by Zhang Qian, All Rights Reserved. 
@Desc    :   Split dataset into trainning and transfer-learning
'''

# here put the import lib
import numpy as np
import os
from sklearn.model_selection import train_test_split

# original
data_2duv = np.load('/data1/qzhang/2duv_small/original/original_dataset.npz')['twoduv']
labels = np.load('/data1/qzhang/2duv_small/original/original_dataset.npz')['labels']
train_data_2duv,transfer_data_2duv,train_label_2duv,transfer_label_2duv = train_test_split(data_2duv,labels,
                                                               stratify = labels,
                                                               test_size=0.1,
                                                               random_state=42)


data_la = np.load('/data1/qzhang/2duv_small/original/original_dataset.npz')['la']
labels = np.load('/data1/qzhang/2duv_small/original/original_dataset.npz')['labels']
train_data_la,transfer_data_la,train_label_la,transfer_label_la = train_test_split(data_la,labels,
                                                               stratify = labels,
                                                               test_size=0.1,
                                                               random_state=42)


data_cd = np.load('/data1/qzhang/2duv_small/original/original_dataset.npz')['cd']
labels = np.load('/data1/qzhang/2duv_small/original/original_dataset.npz')['labels']

train_data_cd,transfer_data_cd,train_label_cd,transfer_label_cd = train_test_split(data_cd,labels,
                                                               stratify = labels,
                                                               test_size=0.1,
                                                               random_state=42)


np.savez('/data1/qzhang/small_dataset_pnas/original/original_dataset.npz',twoduv=train_data_2duv,la=train_data_la,cd=train_data_cd,labels=train_label_2duv)
np.savez('/data1/qzhang/small_dataset_pnas/original/original_transfer_dataset.npz',twoduv=transfer_data_2duv,la=transfer_data_la,cd=transfer_data_cd,labels=transfer_label_2duv)






# homologous
path = '/data1/qzhang/2duv_small/homologous/homologous_dataset.npz'
data_2duv = np.load(path)['twoduv']
labels = np.load(path)['labels']
train_data_2duv,transfer_data_2duv,train_label_2duv,transfer_label_2duv = train_test_split(data_2duv,labels,
                                                               stratify = labels,
                                                               test_size=0.1,
                                                               random_state=42)


data_la = np.load(path)['la']
labels = np.load(path)['labels']
train_data_la,transfer_data_la,train_label_la,transfer_label_la = train_test_split(data_la,labels,
                                                               stratify = labels,
                                                               test_size=0.1,
                                                               random_state=42)


data_cd = np.load(path)['cd']
labels = np.load(path)['labels']

train_data_cd,transfer_data_cd,train_label_cd,transfer_label_cd = train_test_split(data_cd,labels,
                                                               stratify = labels,
                                                               test_size=0.1,
                                                               random_state=42)


np.savez('/data1/qzhang/small_dataset_pnas/homologous/homologous_dataset.npz',twoduv=train_data_2duv,la=train_data_la,cd=train_data_cd,labels=train_label_2duv)
np.savez('/data1/qzhang/small_dataset_pnas/homologous/homologous_transfer_dataset.npz',twoduv=transfer_data_2duv,la=transfer_data_la,cd=transfer_data_cd,labels=transfer_label_2duv)


# nonhomologous
path = '/data1/qzhang/2duv_small/nonhomologous/nonhomologous_dataset.npz'
data_2duv = np.load(path)['twoduv']
labels = np.load(path)['labels']
train_data_2duv,transfer_data_2duv,train_label_2duv,transfer_label_2duv = train_test_split(data_2duv,labels,
                                                               stratify = labels,
                                                               test_size=0.1,
                                                               random_state=42)


data_la = np.load(path)['la']
labels = np.load(path)['labels']
train_data_la,transfer_data_la,train_label_la,transfer_label_la = train_test_split(data_la,labels,
                                                               stratify = labels,
                                                               test_size=0.1,
                                                               random_state=42)


data_cd = np.load(path)['cd']
labels = np.load(path)['labels']

train_data_cd,transfer_data_cd,train_label_cd,transfer_label_cd = train_test_split(data_cd,labels,
                                                               stratify = labels,
                                                               test_size=0.1,
                                                               random_state=42)


np.savez('/data1/qzhang/small_dataset_pnas/nonhomologous/nonhomologous_dataset.npz',twoduv=train_data_2duv,la=train_data_la,cd=train_data_cd,labels=train_label_2duv)
np.savez('/data1/qzhang/small_dataset_pnas/nonhomologous/nonhomologous_transfer_dataset.npz',twoduv=transfer_data_2duv,la=transfer_data_la,cd=transfer_data_cd,labels=transfer_label_2duv)

