#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   broaden.py
@Time    :   2021/05/07 09:00:05
@Author  :   Qian Zhang 
@Contact :   zhangqian.allen@gmail.com
@License :   (C)Copyright 2021
@Desc    :   None
'''

# here put the import lib
import numpy as np
import os 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import sys
import time

start = int(sys.argv[1])
end = int(sys.argv[2])
sigma = int(sys.argv[3])

def spectrum(freq,inten,sigma,x):
    gE=[]
    for Ei in x:
        tot=0
        for Ej,os in zip(freq,inten):
            tot+=os*np.exp(-((((Ej-Ei)/sigma)**2)))
        gE.append(tot)
    return gE


# ahelix_la = np.load('./data/ahelix-la.npy')
ahelix_cd = np.load('./data/ahelix-cd.npy')
# bsheet_la = np.load('./data/bsheet-la.npy')
bsheet_cd = np.load('./data/bsheet-cd.npy')
# otherss_la = np.load('./data/otherss-la.npy')
otherss_cd = np.load('./data/otherss-cd.npy')


sigma = sigma
peak_distance = 70
save_path = './data/broaden_data/cd/sigmma_%d/'%sigma
if not os.path.exists(save_path):
    os.makedirs(save_path)

# la_data = np.concatenate((ahelix_la,bsheet_la,otherss_la),axis=0)[start:end]
cd_data = np.concatenate((ahelix_cd,bsheet_cd,otherss_cd),axis=0)[start:end]
label = np.array([0]*len(ahelix_cd)+[1]*len(bsheet_cd)+[2]*len(otherss_cd))[start:end]


broaden_cd_data = np.zeros((len(cd_data),3000))

tic = time.time()    
for i,data_i in enumerate(cd_data):
    peaks, _ = find_peaks(data_i, distance=peak_distance)
    freq = peaks
    inten = data_i[freq]
    broaden_data_i = spectrum(freq,inten,sigma=sigma,x= range(3000))
    broaden_cd_data[i] = broaden_data_i

np.save(os.path.join(save_path,'cd-data_%d_%d.npy'%(start,end)),broaden_cd_data)
np.save(os.path.join(save_path,'label_%d_%d.npy'%(start,end)),label)
toc = time.time()
print(toc-tic)

