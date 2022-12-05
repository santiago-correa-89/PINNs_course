import os
import vtk
import scipy
from scipy.interpolate import griddata
from pyDOE import lhs

# Plot commands
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotting

# 
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import pickle

from reshapeTest import *
from loadData import *
from PINNs import *
from videoCreater import *

#Pickle save
def mySave(route, variable):    
    with open(route, 'wb') as file:
        pickle.dump(variable, file)
        
#Pickle load
def myLoad(route):    
    with open(route, 'rb') as file:
        variable = pickle.load(file)
    return variable

#OrganizeGrid
def arangeData(x, y, t, T=None):
    if T==None:
        X = x.shape[0]
        Y = y.shape[0]
        T = t.shape[0]
    xGrid = np.c_[np.vstack(np.tile(x[:].repeat(Y), T)), np.transpose(np.vstack(np.tile([y], X*T))), np.vstack([t]).repeat(Y*X)]

    return xGrid

#Evaluate Neural Network on mesh
def evalNN(X, W, b):
    x_tf = tf.convert_to_tensor(X[:, 0:1], dtype=tf.float32)
    y_tf = tf.convert_to_tensor(X[:, 1:2], dtype=tf.float32)
    t_tf = tf.convert_to_tensor(X[:, 2:3], dtype=tf.float32)
    
    predict = net_u(x_tf, y_tf, t_tf, W, b).numpy()
    phi = predict[:, 0:1]
    p = predict[:, 1:2]
    
    return phi, p

# Load Data Xdata refers to spacial position of point, Udata is the Velocity field and Pressure fields for the points. 
W = myLoad('wResult1251242')
b = myLoad('bResult1251242')
lossF = myLoad('lossFResult1251242')
lossU = myLoad('lossDResult1251242')
loss = myLoad('lossResult1251242')

x = np.arange(-5, 15, 0.1)
y = np.arange(-5, 5, 0.05)
t = np.arange(0, 200, 1)

xNorm = myNormalize(x)
yNorm = myNormalize(y)
tNorm = myNormalize(t)

grid = arangeData(x, y, t)
phi, p = evalNN(grid, W, b)

date = str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)

for k in range(t.shape[0]):
    timeEvalMin = k*x.shape[0]*y.shape[0]
    timeEvalMax = timeEvalMin + x.shape[0]*y.shape[0]
    
    grads = np.gradient(phi[timeEvalMin:timeEvalMax].reshape((x.shape[0],y.shape[0])))
    u_ = grads[1]
    v_ = -grads[0]
    p_ = p[timeEvalMin:timeEvalMax].reshape((x.shape[0],y.shape[0]))
    
    fig, ax = plt.subplots()  
    hm = ax.imshow(p_.T, extent=[x.min(), x.max(), y.min(), y.max()])
    animation(hm, r"src/data/fig/presionEstimation/presion", k) 
    plt.close()
    
    fig, ax = plt.subplots()  
    hm = ax.imshow(u_.T, extent=[x.min(), x.max(), y.min(), y.max()])
    animation(hm, r"src/data/fig/uEstimation/u", k) 
    plt.close()
    
    fig, ax = plt.subplots()  
    hm = ax.imshow(v_.T, extent=[x.min(), x.max(), y.min(), y.max()])
    animation(hm, r"src/data/fig/vEstimation/v", k) 
    plt.close()

videoCreater(r"src/data/fig/presionEstimation/presion", r"src/data/fig/presionEstimation/presion" + str(date) + ".avi", t.shape[0])
videoCreater(r"src/data/fig/uEstimation/u", r"src/data/fig/uEstimation/u" + str(date) + ".avi", t.shape[0])
videoCreater(r"src/data/fig/vEstimation/v", r"src/data/fig/vEstimation/v" + str(date) + ".avi", t.shape[0])

for k in range(t.shape[0]):
    os.remove(r"src/data/fig/presionEstimation/presion" + str(k) + ".png")
    os.remove(r"src/data/fig/uEstimation/u" + str(k) + ".png")
    os.remove(r"src/data/fig/vEstimation/v" + str(k) + ".png")

    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lossF, 'r--', lossU, 'bs', loss, 'g^')
ax.set_xlabel('$n iter$')
ax.set_ylabel('Loss')
ax.set_title('Loss evolution', fontsize = 10)
fig.show()