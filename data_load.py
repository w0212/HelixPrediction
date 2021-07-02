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
test_data=np.loadtxt(data_path+"Testset.csv",delimiter=",")

train_imgs=train_data[:,:train_data.shape[1]-5]
test_imgs=test_data[:,:test_data.shape[1]-5]

train_labels=train_data[:,train_data.shape[1]-1]
test_labels=test_data[:,test_data.shape[1]-1]

with open("./Data/Total.pkl","bw") as fh:
    data=(
        train_imgs,
        test_imgs,
        train_labels,
        test_labels
    )

    pickle.dump(data, fh)