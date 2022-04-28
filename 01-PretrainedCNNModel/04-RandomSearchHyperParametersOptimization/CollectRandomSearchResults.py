#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   collect_randomsearch_results.py
@Time    :   2021/05/12 21:19:05
@Author  :   Qian Zhang 
@Contact :   zhangqian.allen@gmail.com
@License :   (C)Copyright 2021
@Desc    :   None
'''

# here put the import lib
import json
import os 
import pandas as pd

if __name__ == '__main__':
    dir_name = './random_search_results/LA-RandomSearchHyperparametersOptimization/'
    drop_out_rate = []
    filters = []
    learning_rate = []
    max_pooling_num = []
    scores = []

    for dir_ in os.listdir(dir_name):
        if dir_.startswith('trial'):
            data = os.path.join(dir_name,dir_,'trial.json')
            with open(data,'r') as f:
                json_file = json.load(f)
            values = json_file['hyperparameters']['values']
            drop_out_rate.append(values['drop_out_rate'])
            filters.append(values['filters'])
            learning_rate.append(values['leraning_rate'])
            max_pooling_num.append(values['max_pooling_num'])
            scores.append(json_file['score'])
            print(json_file.keys())
    df = pd.DataFrame(columns=['drop_out_rate','filters','learning_rate','max_pooling_num','scores'])
    df['drop_out_rate'] = drop_out_rate
    df['filters'] = filters
    df['learning_rate'] = learning_rate
    df['max_pooling_num'] = max_pooling_num
    df['scores'] = scores 
    df.to_csv(os.path.join(dir_name,'results-final.csv'),index=False)
