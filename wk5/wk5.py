# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:13:03 2018

@author: Asamiko
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,mu,sigma):
    x = x.astype(float)
    mu = float(mu)
    sigma = float(sigma)
    var = np.power(sigma,2)
    
    p1 = 1/np.sqrt(2*np.pi*var)
    p2 = np.exp(-1/(2*var) * np.power((x-mu),2))
    return (p1 * p2)

def log_gaussian(x,mu,sigma):
    x = x.astype(float)
    mu = float(mu)
    sigma = float(sigma)
    var = np.power(sigma,2)
    
    term1 = np.log(1/np.sqrt(2*np.pi*var))
    term2 = -1/(2*var) * np.power((x-mu),2)
    p = term1 + term2
    return p


def log_loss(W,X,y):
    L = 0
    for n in range(X.shape[0]): #shape[0] is the amount of samples
        L += np.log( 1 + np.exp(y[n] * np.dot(W.T,X[n])))
    return L

def sig(z):
    return 1 + 1/np.exp(z)

def grad(W,X,y):
    G = 0
    #z = y * W.T * X
    for n in range(X.shape[0]):
        z = y[n] * np.dot(W.T, X[n])
        numer= y[n]*X[n]*np.exp(z)
        denom = 1 + np.exp(z)
        G += numer/denom
        #G += sig(z) * (1-sig(z)) #for some reason doesn't work
    
    return G + np.dot(W.T,W)

def grad_descent(W,X,y,iterations,rate):
    accuracies = []
    W_iterations = []
    
    for i in range(iterations):

        W = W - rate * grad(W, X, y)

        
        #print (i, str(W), log_loss(W, X, y))
        
        y_prob = 1 / (1 + np.exp(-np.dot(X, W)))
                # Threshold at 0.5 (results are 0 and 1)
        y_pred = (y_prob > 0.5).astype(int)
                # Transform [0,1] coding to [-1,1] coding
        y_pred = 2*y_pred - 1

        accuracy = np.mean(y_pred == y)
        accuracies.append(accuracy)
        W_iterations.append(np.copy(W))
        print(W)
        
        
    return W_iterations, accuracies

def plot_path(W, accuracies):
    W = np.array(W)
    plt.figure(figsize = [5,5])
    plt.subplot(211)
    plt.plot(W[:,0], W[:,1], 'ro-')
    plt.xlabel('w$_0$')
    plt.ylabel('w$_1$')
    plt.title('Optimization path')
    
    plt.subplot(212)
    plt.plot(100.0 * np.array(accuracies), linewidth = 2)
    plt.ylabel('Accuracy / %')
    plt.xlabel('Iteration')
    plt.tight_layout()
    #plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")

def main():
    X = np.loadtxt('X.csv',dtype='str',delimiter=',')
    y = np.loadtxt('y.csv', dtype='str',delimiter=',')
    X = X.astype(float)
    y = y.astype(float)
    
    W = np.array([1.0,-1.0])
    step_size = 0.001
    iterations = 200
    
    W_iterations, accuracies = grad_descent(W,X,y,iterations,step_size)
    i = 1
        
    plot_path(W_iterations, accuracies)
    
    
    
main()