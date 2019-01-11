import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_data(X):
    centered = np.add(X, np.negative(np.mean(X, axis = 0)))
    return np.divide(centered, np.std(X, axis = 0))

if __name__ == "__main__":
    #task 1
    data = np.loadtxt("locationData.csv")
    sz = np.shape(data)
    print(sz)
    
    #task 2
    #plt.plot(data[:,0], data[:,1])
    ax = plt.subplot(1, 1, 1, projection = "3d")
    plt.plot(data[:,0], data[:,1], data[:,2])

    #task 3    
    data_norm = normalize_data(data)
    
    print(np.mean(data_norm, axis = 0)) # Values very close to 0
    print(np.std(data_norm, axis = 0)) # Values are 1