#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File : Lorentzian-LA.py
@Time : 2021/09/01 11:28:18
@Author : Wang Zhengjie 
@Version : 1.0
@Contact : wangzhengjie.CN@gmail.com
@License : (C)Copyright 2020-2021, MTSD, China University of Petroleum (East China)
@Desc : None
'''

# here put the import lib
import numpy as np
import os 
import matplotlib.pyplot as plt
import sys
import time

start = int(sys.argv[1])
end = int(sys.argv[2])
gamma = int(sys.argv[3])

def Lorentzian(x,y,gamma):
    Y = np.zeros(len(x))
    for x0, y0 in zip(x,y):
        Y += gamma / 2. / np.pi / ((x - x0)**2 + 0.25 * gamma ** 2) * y0
    return Y

ahelix_la = np.load('./data/ahelix-la.npy')
# ahelix_cd = np.load('./data/ahelix-cd.npy')
bsheet_la = np.load('./data/bsheet-la.npy')
# bsheet_cd = np.load('./data/bsheet-cd.npy')
otherss_la = np.load('./data/otherss-la.npy')
# otherss_cd = np.load('./data/otherss-cd.npy')


gamma = gamma
# peak_distance = 70
save_path = './data/broaden_data/la/gamma_%d/'%gamma
if not os.path.exists(save_path):
    os.makedirs(save_path)

la_data = np.concatenate((ahelix_la,bsheet_la,otherss_la),axis=0)[start:end]
# cd_data = np.concatenate((ahelix_cd,bsheet_cd,otherss_cd),axis=0)[start:end]
label = np.array([0]*len(ahelix_la)+[1]*len(bsheet_la)+[2]*len(otherss_la))
# label = np.array([0]*len(ahelix_cd)+[1]*len(bsheet_cd)+[2]*len(otherss_cd))[start:end]
print(label.shape)
label = label[start:end]

data = la_data[:,1000:3000].copy()
x=np.arange(40000,60000,10)

broaden_data = np.zeros((len(data),2000))

tic = time.time()
for i,data_i in enumerate(data):
    broaden_data_i = Lorentzian(x=x,y=data_i,gamma=gamma)
    broaden_data[i] = broaden_data_i

np.save(os.path.join(save_path,'la-data_%d_%d.npy'%(start,end)),broaden_data)
np.save(os.path.join(save_path,'label_%d_%d.npy'%(start,end)),label)
toc = time.time()
print(toc-tic)

