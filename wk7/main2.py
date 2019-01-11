# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:10:05 2018

@author: Konsta Peltoniemi
"""

from sklearn.feature_selection import RFECV
from scipy.io import loadmat
import numpy as np
from matplotlib.pyplot import plot as plt

rfe = RFECV(step = 50,verbose = 1)
X_train_r = rfe.fit(X_train,y_train)
print(np.shape(rfe.support_))
plt.plot(range(0,10001,50), rfe.grid_scores_)