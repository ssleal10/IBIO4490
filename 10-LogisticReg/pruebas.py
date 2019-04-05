# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:00:46 2019

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt

def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013/fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        #emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_data()


gt = [1,2,3,4]
pred = [2,2,3,4]

where = np.where(gt == np.amax(gt))
print(where[0][0]+1)
#para los labels
gt1 = [x+1 for x in gt]
out = []
out.append(gt)
out.append(pred)

params = 48*48
out = (1,2) # smile label
lr = 0.003 # Change if you want
W = np.random.randn(params, out)
b = np.random.randn(out)

check = np.log(softmax(pred))
what = -np.multiply(np.log(softmax(pred)),gt)