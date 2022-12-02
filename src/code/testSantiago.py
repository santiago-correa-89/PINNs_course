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
def evalNNesc(X, W, b):
    x_tf = tf.convert_to_tensor(X[:, 0:1], dtype=tf.float32)
    y_tf = tf.convert_to_tensor(X[:, 1:2], dtype=tf.float32)
    t_tf = tf.convert_to_tensor(X[:, 2:3], dtype=tf.float32)
    
    predict = net_u(x_tf, y_tf, t_tf, W, b).numpy()
    phi = predict[:, 0:1]
    p = predict[:, 1:2]
    
    return phi, p

W = myLoad('wResult1212048')
b = myLoad('bResult1212048')
loss = myLoad('lossResult1212048')

x = np.arange(-5, 15, 0.1)
y = np.arange(-5,5, 0.1)
t = np.arange(0, 200, 1)

grid = arangeData(x, y, t)
phi, p = evalNNesc(grid, W, b)

fecha = str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
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

videoCreater(r"src/data/fig/presionEstimation/presion", r"src/data/fig/presionEstimation/presion.avi", t.shape[0])
videoCreater(r"src/data/fig/uEstimation/u", r"src/data/fig/uEstimation/u.avi", t.shape[0])
videoCreater(r"src/data/fig/vEstimation/v", r"src/data/fig/vEstimation/v.avi", t.shape[0])

for k in range(t.shape[0]):
    os.remove(r"src/data/fig/presionEstimation/presion" + str(k) + ".png")
    os.remove(r"src/data/fig/uEstimation/u" + str(k) + ".png")
    os.remove(r"src/data/fig/vEstimation/v" + str(k) + ".png")
    
    

print('prueba')

# lossnp = tf.concat(loss,0).numpy()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(lossnp)
# ax.set_xlabel('$n iter$')
# ax.set_ylabel('$loss$')    
# ax.set_title('Loss evolution', fontsize = 10)