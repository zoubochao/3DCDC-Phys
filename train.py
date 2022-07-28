# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from model import CAN_3D
from inference_preprocess import preprocess_raw_video,detrend
import random
import h5py
import scipy
from scipy.signal import butter
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import losses as LOSSES
import tensorflow as tf
from scipy.signal import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


n_frame       = 10
nb_filters1   = 32 
nb_filters2   = 64
input_shape_1 = (36,36,n_frame,3)    #用于3D卷积模型
input_shape_2 = (36,36,3)            
kernel_size   = (3, 3)
dropout_rate1 = 0.25
dropout_rate2 = 0.5
pool_size     = (2, 2)
nb_dense      = 128
lr            = 1.0
batch_size1    = 1        #随机抽取的视频数量多少，太大会造成内存溢出
epochs        = 1000
steps_per_epoch = 10      #每个epoch训练的次数
dataset_type  = 1        #用哪个数据集训练，1为用hdf5文件保存标签的数据集；2为用txt保存类标签的数据集
train_path    = './train.txt'    #保存视频文件路径的txt，具体格式参考下面注释

test_path     = './train.txt'
dataset_path  = 'E:/dataset/cohface/cohface/cohface/'  #视频文件路径
sample_list   = None     #保存样本路径：['D:/data/1/0/data','D:/data/1/1/data',...]
start         = 0
end           = 5000
cv            = 10        #k折交叉验证

with open(train_path) as f:
    sample_list = [dataset_path+i.replace('\n','') for i in f.readlines()]

#获取第二种数据集的标签
def get_label(label_txt):
    with open(label_txt) as f:
        a = f.readlines()
    b = a[0].replace('\n','').split(' ')
    c = [i for i in b if i!='']
    d = []
    for i in c:
        l = float(i.split('e')[0])
        r = int(i.split('e')[1])
        if r!=0:
            l = l*(10**r)
        d.append(l)
    return np.array(d,dtype=np.float32)


def gen(video_path):
    data ,label_y ,label_r = [],[],[]
    for temp_path in video_path:
        try:
            dXsub = preprocess_raw_video(temp_path+'.avi')
            f1 = h5py.File(temp_path+'.hdf5', 'r')
            
            dXsub_len = (dXsub.shape[0] // n_frame)  * n_frame
            dXsub_len = dXsub.shape[0] 
            dysub = np.array(f1['pulse'])       #脉搏、心率

            dysub = resample(dysub,dXsub_len)
            dysub = (dysub-np.mean(dysub))/np.std(dysub)



            # drsub = np.array(f1['respiration']) #呼吸
            #         #drsub = drsub[:dXsub_len]
            # drsub = resample(drsub,dXsub_len)
            # drsub = (drsub-np.mean(drsub))/np.std(drsub)
            #num_window = 22#dXsub.shape[0] - (n_frame + 1)
            num_window = dXsub.shape[0] - (n_frame + 1)
            
            #dysub = np.array(f1['pulse'])       #脉搏、心率
            #dysub = dysub[:dXsub_len]
            #dysub = resample(dysub,dXsub_len)
            
            # print("dXsub.shape",dXsub.shape)
            # print("dysub.shape",dysub.shape)

            #dysub = (dysub-np.mean(dysub))/np.std(dysub)
            for f in range(num_window):
                data.append(dXsub[f:f+n_frame,:,:,:])
            
                #label_y.append([dysub[f:f+n_frame] for f in range(num_window)])
                label_y.append(dysub[f:f+n_frame])
        except Exception as e:
            continue

    data = np.array(data).reshape((-1,n_frame,36,36,6))
    data = np.swapaxes(data, 1, 3) # (-1, 36, 36, 10, 6)
    data = np.swapaxes(data, 1, 2) # (-1, 36, 36, 10, 6)
    #label_y = np.array(label_y).reshape(-1,n_frame)
    #print("data",data.shape)
    label_y = np.array(label_y).reshape(-1,n_frame)
    #print("label_y",label_y.shape)
    output = (data[:,:,:,:,:3],data[:,:,:,:,-3:])
    #label = (label_y, label_r)
    return output,label_y

kf = KFold(n_splits=cv)
flag = 1
His = {'val_mae':[]}

for train_index,vali_index in kf.split(sample_list):
    train_sample = [sample_list[i] for i in train_index]
    vali_sample  = [sample_list[i] for i in vali_index]

    # print("train_sample",train_sample)
    # print("vali_sample",vali_sample)
    
    output_tr,label_tr = gen(train_sample)
    output_va,label_va = gen(vali_sample)
    
    
    Model = CAN_3D(n_frame,nb_filters1,nb_filters2,input_shape_1)
    optimizer = optimizers.Adadelta(learning_rate=lr)
    Model.compile(loss="huber_loss", optimizer=optimizer,metrics=['mae','mse'])

    checkpoint = ModelCheckpoint(filepath='./can_3d-cv'+str(flag)+'-{epoch:02d}-{loss:.2f}.hdf5',\
                                 monitor='val_loss',
                                 save_best_only=False,
                                 save_weights_only=True)
    history = Model.fit(output_tr,label_tr,batch_size=12,epochs=50,verbose=1,\
                        callbacks=[checkpoint],validation_data=(output_va,label_va))

    His['val_mae'].append(history.history['val_mae'])
    flag+=1


