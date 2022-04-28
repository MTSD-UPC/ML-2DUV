#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   LACD.py
@Time    :   2021/04/29 22:12:43
@Author  :   Qian Zhang 
@Contact :   zhangqian.allen@gmail.com
@License :   (C)Copyright 2021
@Desc    :   shell command: nohup python -u 01-TransferLeanringRandomSearchExample.py >> RandomSearch-LA.log 2>&1 &
'''

# here put the import lib
import numpy as np
import tensorflow as tf
import os 
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = "0" #use GPU:0
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable=True)
    
# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"]) # parallel computation

print(gpus)


####################### CONFIGUE ##################################################
# def. some functions 

# freeze some layers of base model
def freeze_layer(base_model,layer_name = []):
    
    base_model.trainable=True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name in layer_name:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    return base_model

def build_model(hp):
    '''
    Pars:
        hidden_layers_num: type:list
        drop_out: rate of drop out layer
        lr: learning rate
    '''
    filters1 = hp.Choice('filters',values=[128,256,512],default=128)
    filters2 = hp.Choice('filters',values=[32,64,128],default=64)
    LR = hp.Choice('leraning_rate',values=[0.01,0.001,0.002,0.008,0.004,0.0001,0.0004,0.0008],default=0.001)
    dropu_out_rate = hp.Float('drop_out_rate',min_value=0.1,max_value=0.5,step=0.1,default=0.2)
    
    new_model = tf.keras.models.Sequential()
    new_model.add(base_model)
    new_model.add(tf.keras.layers.Flatten())
    new_model.add(tf.keras.layers.Dropout(dropu_out_rate))
    # for num_ in hidden_layers_num:
    #     new_model.add(tf.keras.layers.Dense(num_,activation='relu'))
    new_model.add(tf.keras.layers.Dense(filters1,activation='relu'))
    new_model.add(tf.keras.layers.Dense(filters2,activation='relu'))
    new_model.add(tf.keras.layers.Dense(3,activation='softmax'))
    new_model.compile(loss = 'sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=LR), metrics=['acc'])
    return new_model



# def build_model(hp):
#     filters = hp.Choice('filters',values=[32,64,128,256,512],default=256)
#     maxpooling_num = hp.Int('max_pooling_num',min_value =2,max_value=20,default=12,step=2)
#     LR = hp.Choice('leraning_rate',values=[0.01,0.001,0.002,0.008,0.004,0.0001,0.0004,0.0008],default=0.001)
#     dropu_out_rate = hp.Float('drop_out_rate',min_value=0.1,max_value=0.5,step=0.1,default=0.2)
    
    
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv1D(filters=filters,
#                                kernel_size= 10,
#                                activation='relu',
#                                input_shape=(input_size,1),
#                                padding='same'),
#         tf.keras.layers.MaxPooling1D(pool_size=maxpooling_num),
#         tf.keras.layers.Conv1D(filters=filters,
#                                 kernel_size= 10,
#                                 activation='relu',
#                                 padding='same'),
#         tf.keras.layers.MaxPooling1D(pool_size=maxpooling_num),
#         # tf.keras.layers.Conv1D(filters=filters,
#         #                         kernel_size= 10,
#         #                         activation='relu',
#         #                         padding='same'),
#         # tf.keras.layers.MaxPooling1D(pool_size=maxpooling_num),
#         # tf.keras.layers.Conv1D(filters=32,
#         #                        kernel_size=10,
#         #                        activation='relu',
#         #                        padding='same'),
#         # tf.keras.layers.MaxPooling1D(12),
#         tf.keras.layers.Dropout(dropu_out_rate),
#         # tf.keras.layers.LSTM(128,return_sequences=True),
#         # tf.keras.layers.LSTM(128),
#         # tf.keras.layers.Dropout(0.4),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128,activation='relu'),
#         tf.keras.layers.Dense(64,activation='relu'),
#         tf.keras.layers.Dropout(dropu_out_rate),
#         tf.keras.layers.Dense(3,activation='softmax')
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR),loss='sparse_categorical_crossentropy',
#                     metrics=['acc'])
#     return model
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
    # batch_size_per_replica=32  # batch size of per GPU 
    batch_size = 32
    directory = 'random_search_results'
    # project_name = 'CNNlayers-3'
    project_name = 'LA-01-TransferLeanringRandomSearchExample'
    max_trials = 20 # how many times
    EPOCH= 20      # 

    # load original data to get test data
    data_la = np.load('/data1/qzhang/small_dataset_pnas/original/original_dataset.npz')['la']
    labels = np.load('/data1/qzhang/small_dataset_pnas/original/original_dataset.npz')['labels']
    
    # preprocess
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    labels_cat = encoder.fit_transform(labels)
    print(encoder.classes_)
    print(np.unique(labels_cat))

    data_la = minmax_scale(data_la,axis=1).reshape(-1,input_size,1)
    
    from sklearn.model_selection import train_test_split
    train_data,test_data,train_label,test_label = train_test_split(data_la.reshape(-1,3000,1),labels_cat,
                                                                stratify = labels_cat,
                                                                test_size=0.2,
                                                                random_state=42)
        
    
    # load tranfer learning data
    original_transfer_la = np.load('/data1/qzhang/small_dataset_pnas/original/original_transfer_dataset.npz')['la']
    original_transfer_labels = np.load('/data1/qzhang/small_dataset_pnas/original/original_transfer_dataset.npz')['labels']

    homo_transfer_la = np.load('/data1/qzhang/small_dataset_pnas/homologous/homologous_transfer_dataset.npz')['la']
    homo_transfer_labels = np.load('/data1/qzhang/small_dataset_pnas/homologous/homologous_transfer_dataset.npz')['labels']


    nonhomo_transfer_la = np.load('/data1/qzhang/small_dataset_pnas/nonhomologous/nonhomologous_transfer_dataset.npz')['la']
    nonhomo_transfer_labels = np.load('/data1/qzhang/small_dataset_pnas/nonhomologous/nonhomologous_transfer_dataset.npz')['labels']

    transfer_data = np.concatenate((original_transfer_la,homo_transfer_la,nonhomo_transfer_la))
    transfer_label = np.concatenate((original_transfer_labels,homo_transfer_labels,nonhomo_transfer_labels))

    # process labels
    transfer_label_cat = encoder.fit_transform(transfer_label)

    # preprocess data
    from sklearn.preprocessing import minmax_scale
    transfer_data = minmax_scale(transfer_data,axis=1).reshape(-1,3000,1)

    print('Transfer data size: %d, which includes \n original:%d, homo:%d, non-homo:%d'%(len(transfer_data),len(original_transfer_la),
                                                                            len(homo_transfer_la),
                                                                            len(nonhomo_transfer_la)))

    # split transfer data into train_data  and val data
    train_x,val_x,train_y,val_y = train_test_split(transfer_data,transfer_label_cat,stratify=transfer_label_cat,test_size = 0.2,random_state=42)


    # load base model
    raw_model = tf.keras.models.load_model('../../01-PretrainedCNNModel/1DCNNDemoModel.h5')
    # we just need top 6 layers
    base_model = tf.keras.models.Sequential(raw_model.layers[:6])
    
    # #  freeze layer
    # base_model = freeze_layer(base_model=base_model)
    # build a fresh CNN-FCNN model

    
    tuner =  RandomSearch(
                        hypermodel=build_model,
                        max_trials=max_trials, #参数组合的数量，及训练模型的数量
                        objective='val_acc',
                        executions_per_trial=1,
                        # distribution_strategy=strategy,
                        directory=directory,
                        project_name = project_name
                        )
    print(tuner.search_space_summary())
    # model= build_model()
    
    # early stop configuration
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc',patience=5) # Number of epochs with no improvement after which training will be stopped

    # batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    
    # tuner.search(X_train, y_train, epochs=EPOCH,batch_size=batch_size,callbacks=[callback])
    tuner.search(train_x, train_y, validation_data=(val_x,val_y), epochs=EPOCH,batch_size=batch_size,callbacks=[callback],verbose=2)
    best_model = tuner.get_best_models(num_models=1)[0]

    print(tuner.results_summary())

    print('Best model structure:',best_model.summary())
    loss, accuracy = best_model.evaluate(test_data,test_label,verbose=2)
    print('Best model test results:',accuracy)
    best_model.save(os.path.join(directory,project_name,'best_model.h5'))

