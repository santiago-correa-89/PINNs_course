import tensorflow as tf
import numpy as np
import time
import datetime
import os
import matplotlib.pyplot as plt

from utilities import *
from train import *

def evalNN(X, W, b):
    x_tf = X[:, 0:1]
    y_tf = X[:, 1:2]
    t_tf = X[:, 2:3]
    
    predict = net_u(x_tf, y_tf, t_tf, W, b).numpy()
    phi = predict[:, 0:1]
    p = predict[:, 1:2]
    
    return phi, p

# Load Data Xdata refers to spacial position of point, Udata is the Velocity field and Pressure fields for the points. 
W = myLoad('wResult1261527')
b = myLoad('bResult1261527')

x = np.arange(-5, 15, 0.1)
y = np.arange(-5, 5, 0.05)
t = np.arange(0, 2, 0.01)

grid = arangeData(x, y, t)
grid_tf = tf.convert_to_tensor(grid, dtype=tf.float32)
phiPred, pPred = evalNN(grid_tf, W, b)

date = str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)

for k in range(t.shape[0]):
    
    timeEvalMin = k*x.shape[0]*y.shape[0]
    timeEvalMax = timeEvalMin + x.shape[0]*y.shape[0]
    
    grads = np.gradient(phiPred[timeEvalMin:timeEvalMax].reshape((x.shape[0],y.shape[0])))
    u = grads[1]
    v = -grads[0]
    p = pPred[timeEvalMin:timeEvalMax].reshape((x.shape[0],y.shape[0]))
    
    fig, ax = plt.subplots()  
    hm = ax.imshow(p.T, extent=[x.min(), x.max(), y.min(), y.max()])
    animation(hm, r"src/data/fig/presionEstimation/presion", k, 'Pressure Field') 
    plt.close()
    
    fig, ax = plt.subplots()  
    hm = ax.imshow(u.T, extent=[x.min(), x.max(), y.min(), y.max()])
    animation(hm, r"src/data/fig/uEstimation/u", k, 'U Field') 
    plt.close()
    
    fig, ax = plt.subplots()  
    hm = ax.imshow(v.T, extent=[x.min(), x.max(), y.min(), y.max()])
    animation(hm, r"src/data/fig/vEstimation/v", k, 'V Field') 
    plt.close()

videoCreater(r"src/data/fig/presionEstimation/presion", r"src/data/fig/presionEstimation/presion" + str(date) + ".avi", t.shape[0])
videoCreater(r"src/data/fig/uEstimation/u", r"src/data/fig/uEstimation/u" + str(date) + ".avi", t.shape[0])
videoCreater(r"src/data/fig/vEstimation/v", r"src/data/fig/vEstimation/v" + str(date) + ".avi", t.shape[0])

for k in range(t.shape[0]):
    os.remove(r"src/data/fig/presionEstimation/presion" + str(k) + ".png")
    os.remove(r"src/data/fig/uEstimation/u" + str(k) + ".png")
    os.remove(r"src/data/fig/vEstimation/v" + str(k) + ".png")