import tensorflow as tf
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os

from utilities import *

def Processing(x, y, t, estimation, loss, lossF, lossU, folder, date):
    
    phi = estimation[:, 0:1].numpy()
    p = estimation[:, 1:2].numpy()

    #u = estimation[:, 0:1].numpy()
    #v = estimation[:, 1:2].numpy()
    #p = estimation[:, 1:2].numpy()

    for k in range(t.shape[0]):
    
        timeEvalMin = k*x.shape[0]*y.shape[0]
        timeEvalMax = timeEvalMin + x.shape[0]*y.shape[0]

        grads = np.gradient(phi[timeEvalMin:timeEvalMax].reshape((x.shape[0],y.shape[0])))
        u = grads[1]
        v = -grads[0]
    
        fig, ax = plt.subplots()  
        hm = ax.imshow(p[timeEvalMin:timeEvalMax].reshape((x.shape[0],y.shape[0])).T, extent=[x.min(), x.max(), y.min(), y.max()])
        animation(hm, folder + '/pEstimation/p', k, 'Pressure Field') 
        plt.close()
    
        fig, ax = plt.subplots()  
        #hm = ax.imshow(u[timeEvalMin:timeEvalMax].reshape((x.shape[0],y.shape[0])).T, extent=[x.min(), x.max(), y.min(), y.max()])
        hm = ax.imshow(u.T, extent=[x.min(), x.max(), y.min(), y.max()])
        animation(hm, folder + '/uEstimation/u', k, 'U Field') 
        plt.close()
        
        fig, ax = plt.subplots()  
        #hm = ax.imshow(v[timeEvalMin:timeEvalMax].reshape((x.shape[0],y.shape[0])).T, extent=[x.min(), x.max(), y.min(), y.max()])
        hm = ax.imshow(v.T, extent=[x.min(), x.max(), y.min(), y.max()])
        animation(hm, folder + '/vEstimation/v', k, 'V Field') 
        plt.close()

    videoCreater(folder + '/pEstimation/p', folder + '/pEstimation/p' + str(date) + ".avi", t.shape[0])
    videoCreater(folder + '/uEstimation/u', folder + '/uEstimation/u' + str(date) + ".avi", t.shape[0])
    videoCreater(folder + '/vEstimation/v', folder + '/vEstimation/v' + str(date) + ".avi", t.shape[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss, 'bs')
    ax.set_xlabel('$n iter$')
    ax.set_ylabel('Loss')
    plt.yscale('log')
    ax.set_title('Loss Evolution', fontsize = 10)
    fig.savefig(folder + '/LossEstimation.png', dpi=200)
    fig.show() 
    plt.close()