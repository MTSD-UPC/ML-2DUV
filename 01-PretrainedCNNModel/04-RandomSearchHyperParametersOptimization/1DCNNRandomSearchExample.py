#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LACD.py
@Time    :   2021/04/29 22:12:43
@Author  :   Qian Zhang 
@Contact :   zhangqian.allen@gmail.com
@License :   (C)Copyright 2021
@Desc    :   shell command: nohup python -u 1DCNNRandomSearchExample.py >> RandomSearch-LA.log 2>&1 &
'''

# here put the import lib
import numpy as np
import tensorflow as tf
import os 
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" #use GPU:0
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable=True)
    
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"]) # parallel computation

print(gpus)


####################### CONFIGUE ##################################################
    
def build_model(hp):
    filters = hp.Choice('filters',values=[32,64,128,256,512],default=256)
    maxpooling_num = hp.Int('max_pooling_num',min_value =2,max_value=20,default=12,step=2)
    LR = hp.Choice('leraning_rate',values=[0.01,0.001,0.002,0.008,0.004,0.0001,0.0004,0.0008],default=0.001)
    dropu_out_rate = hp.Float('drop_out_rate',min_value=0.1,max_value=0.5,step=0.1,default=0.2)
    
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=filters,
                               kernel_size= 10,
                               activation='relu',
                               input_shape=(input_size,1),
                               padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=maxpooling_num),
        tf.keras.layers.Conv1D(filters=filters,
                                kernel_size= 10,
                                activation='relu',
                                padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=maxpooling_num),
        # tf.keras.layers.Conv1D(filters=filters,
        #                         kernel_size= 10,
        #                         activation='relu',
        #                         padding='same'),
        # tf.keras.layers.MaxPooling1D(pool_size=maxpooling_num),
        # tf.keras.layers.Conv1D(filters=32,
        #                        kernel_size=10,
        #                        activation='relu',
        #                        padding='same'),
        # tf.keras.layers.MaxPooling1D(12),
        tf.keras.layers.Dropout(dropu_out_rate),
        # tf.keras.layers.LSTM(128,return_sequences=True),
        # tf.keras.layers.LSTM(128),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(dropu_out_rate),
        tf.keras.layers.Dense(3,activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR),loss='sparse_categorical_crossentropy',
                    metrics=['acc'])
    return model
#######################################################################################################################


# import kerastuner
# import numpy as np
# from sklearn import model_selection
# class CVTuner(kerastuner.engine.tuner.Tuner):
#     def run_trial(self, trial, x, y, callbacks,cv_num=5,batch_size=32, epochs=1):
#         cv = model_selection.KFold(cv_num)
#         val_acc = []
#         for train_indices, test_indices in cv.split(x):
#             x_train, x_test = x[train_indices], x[test_indices]
#             y_train, y_test = y[train_indices], y[test_indices]
#             model = self.hypermodel.build(trial.hyperparameters)
#             model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=batch_size, epochs=epochs,callbacks=[callbacks])
#             val_acc.append(model.evaluate(x_test, y_test))
#         self.oracle.update_trial(trial.trial_id, {'val_acc': np.mean(val_acc)})
#         self.save_model(trial.trial_id, model)

# -------------------------------
if __name__=='__main__':

#--------------------- CONFIGURATION -------------------------------------------------#
    input_size = 3000   # spectral size
    batch_size_per_replica=64  # batch size of per GPU 
    directory = 'random_search_results'
    # project_name = 'CNNlayers-3'
    project_name = 'LA-RandomSearchHyperparametersOptimization'
    max_trials = 20 # how many times
    EPOCH= 20      # 

    # load data
    la_data = np.load('/data1/qzhang/small_dataset_pnas/original/original_dataset.npz')['la']
    labels = np.load('/data1/qzhang/small_dataset_pnas/original/original_dataset.npz')['labels']
    
    # preprocess
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    labels_cat = encoder.fit_transform(labels)
    print(encoder.classes_)
    print(np.unique(labels_cat))

    la_data = minmax_scale(la_data,axis=1).reshape(-1,input_size,1)

    # for la model 
    X_train, X_test, y_train, y_test = train_test_split(la_data,labels_cat,stratify=labels_cat,test_size = 0.2,random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,stratify=y_train,test_size = 0.2,random_state = 42)
    # np.save('./X_test.npy',X_test)
    # np.save('./y_test.npy',y_test)
    # fro cd model 
    # X_train, X_test, y_train, y_test = train_test_split(cd_data,label,stratify=label,test_size = 0.2,random_state = 42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,stratify=y_train,test_size = 0.2,random_state = 42)

    # cross validation    
    # tuner = CVTuner(
    #   hypermodel=build_model,
    #   oracle=kerastuner.oracles.RandomSearch(
    #     objective='val_acc',
    #     max_trials=3),
    #     distribution_strategy=strategy,
    #     directory='my_dir',
    #     project_name = 'CD-random-search'
    #   )

    tuner =  RandomSearch(
                        hypermodel=build_model,
                        max_trials=max_trials, #参数组合的数量，及训练模型的数量
                        objective='val_acc',
                        executions_per_trial=1,
                        distribution_strategy=strategy,
                        directory=directory,
                        project_name = project_name
                        )
    print(tuner.search_space_summary())
    # model= build_model()
    
    # early stop configuration
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience=5) # Number of epochs with no improvement after which training will be stopped

    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    
    # tuner.search(X_train, y_train, epochs=EPOCH,batch_size=batch_size,callbacks=[callback])
    tuner.search(X_train, y_train, validation_data=(X_val,y_val), epochs=EPOCH,batch_size=batch_size,callbacks=[callback],verbose=2)
    best_model = tuner.get_best_models(num_models=1)[0]

    print(tuner.results_summary())

    print('Best model structure:',best_model.summary())
    loss, accuracy = best_model.evaluate(X_test,y_test,verbose=2)
    print('Best model test results:',accuracy)
    best_model.save(os.path.join(directory,project_name,'best_model.h5'))

