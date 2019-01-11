# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:11:52 2018

@author: asamiko
"""

import glob
import numpy as np
from matplotlib.image import imread
from skimage.feature import local_binary_pattern

#Loads the German Traffic Sign Recognition Benchmark dataset and extracts
#features using Local Binary Pattern algorithm
def load_GTSRB():
    Class1 = []
    Class2 = []
    GT = [] #ground truth
    
    C1files = glob.glob(r'''C:\ml\GTSRB_subset\class1\*.jpg''' )
    C2files = glob.glob(r'''C:\ml\GTSRB_subset\class2\*.jpg''' )
    
    radius = 5
    n_points = 8
    
    for file in C1files:
        img = imread(file)
        lbp = local_binary_pattern(img, n_points, radius)
        hist = np.histogram(lbp, bins = range(0,256))[0]
        Class1.append(hist)
        GT.append(0)
    
    for file in C2files:
        img = imread(file)
        lbp = local_binary_pattern(img, n_points, radius)
        hist = np.histogram(lbp, bins = range(0,256))[0]
        Class2.append(hist)
        GT.append(1)
    
    X = np.concatenate((Class1,Class2),axis=0)
    return X,GT


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.ensemble import *

#Experiment with regularization and distance metrics for the classifiers
task4 = False
#Try out 4 different tree classifiers for the classification task
task5 = True

if (task4):    
    X, GT = load_GTSRB()
    X = normalize(X)
    X_train,X_test,y_train,y_test = train_test_split(X,GT)
    
    
    clf_list = [LogisticRegression(),SVC()]
    clf_name = ['LR','SVC']
    
    C_range = 10.0**np.arange(-5,1)
    scores = []
    clf_scores = []
    
    for clf,name in zip(clf_list, clf_name):
        for C in C_range:
                for penalty in ["l1", "l2"]:
                    clf.C = C
                    clf.penalty = penalty
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    scores.append(score)
                    clf_scores.append((name,C,penalty,score))                
        
    i = scores.index(max(scores))
    print(clf_scores[i])
    
if(task5):
    X, GT = load_GTSRB()
    X = normalize(X)
    X_train,X_test,y_train,y_test = train_test_split(X,GT)
    
    clf_list = [RandomForestClassifier(),ExtraTreesClassifier(), \
                AdaBoostClassifier(),GradientBoostingClassifier()]
    clf_name = ['Rand Forest','Extreme Forest','Ada','Boosted Trees']
    scores = []
    
    for clf,name in zip(clf_list,clf_name):
        clf.n_esitmators = 100
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        scores.append(score)
    
    i = scores.index(max(scores))
    print(clf_name[i], max(scores))