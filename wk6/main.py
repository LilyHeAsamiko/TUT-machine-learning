# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:53:13 2018

@author: Asamiko
"""

import numpy as np
import glob
from matplotlib.image import imread
from keras.preprocessing import image
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import losses

def load_GTSRB():
    Class1 = []
    Class2 = []
    imlist = []
    GT = [] #ground truth
    
    C1files = glob.glob(r'''C:\ml\GTSRB_subset_2\class1\*.jpg''' )
    C2files = glob.glob(r'''C:\ml\GTSRB_subset_2\class2\*.jpg''' )
    
    for file in C1files:
        imlist.append(np.array(image.load_img(file)))
        GT.append(0)
    
    for file in C2files:
        imlist.append(np.array(image.load_img(file)))
        GT.append(1)
    
    return np.asarray(imlist),GT

if __name__ == "__main__":
    X,Y = load_GTSRB()
    X = X.astype(float)
    X_norm = np.ones_like(X)
    X_norm = X_norm.astype(float)
    for i in range (np.shape(X)[0]):
        X_norm[i] = np.divide((X[i,:,:,:] - np.min(X[i,:,:,:])), float(np.max(X[i,:,:,:])))
    
    X_norm = np.array(X_norm)
    Y = np.array(Y)
    Y = np_utils.to_categorical(Y,2)
    X_train, X_test, y_train,y_test = train_test_split(X_norm,Y,test_size=0.2)
    
    model = Sequential()
    N = 32
    w,h = 5, 5
    model.add(Conv2D(N, (w, h),
                input_shape=(64, 64, 3),
                activation = 'relu',
                padding = 'same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(N, (w, h),
                activation = 'relu',
                padding = 'same'))
    model.add(MaxPooling2D((4,4)))
    model.add(Flatten())
    model.add(Dense(100, activation = 'sigmoid'))
    model.add(Dense(2, activation = 'sigmoid'))
    print(model.summary())
    
    model.compile(optimizer='sgd',loss='binary_crossentropy',metrics = ['accuracy'])
    model.fit(X_train,y_train,batch_size=32,epochs=32,validation_data=[X_test,y_test])
              
    