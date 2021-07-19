# -*- coding: utf-8 -*-
'''
@Time    : 2021/6/30 13:45
@Author  : Junfei Sun
@Email   : sjf2002@sohu.com
@File    : data_load.py
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

data_path="./Data/"
train_data=np.loadtxt(data_path+"Dataset.csv",delimiter=",")
predict_data=np.loadtxt("./Predict/Predictset.csv",delimiter=",")

train_imgs=train_data[:,:train_data.shape[1]-5]
predict_imgs=predict_data[:,:predict_data.shape[1]-5]

train_labels=train_data[:,train_data.shape[1]-1]
predict_labels=predict_data[:,predict_data.shape[1]-1]

with open("./Data/Total.pkl","bw") as fh:
    data=(
        train_imgs,
        predict_imgs,
        train_labels,
        predict_labels
    )



    pickle.dump(data, fh)