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
        Y = y.shape[0]
        X = x.shape[0]
        T = t.shape[0]
    xGrid = np.c_[np.vstack([x]).repeat(Y*T), np.transpose(np.vstack(np.tile([y], X*T))), np.vstack(t[:].repeat(Y*X))]

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

W = myLoad('W_result')
b = myLoad('b_result')
loss = myLoad('loss_result')

x = np.arange(-5, 15, 0.5)
y = np.arange(-5,5, 0.25)
t = np.arange(0, 200, 1)

grid = arangeData(x, y, t)

phi, p = evalNNesc(grid, W, b)

Xmesh, Ymesh = np.meshgrid(x, y)            
time = 40
timeEvalMin = time*x.shape[0]*y.shape[0]
timeEvalMax = timeEvalMin + x.shape[0]*y.shape[0] - 1
fig, ax = plt.subplots()
hm = ax.imshow(p[timeEvalMin:timeEvalMax:1], extent=[Xmesh.min(), Xmesh.max(), Ymesh.max(), Ymesh.min()]) 


lossnp = tf.concat(loss,0).numpy()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lossnp)
ax.set_xlabel('$n iter$')
ax.set_ylabel('$loss$')    
ax.set_title('Loss evolution', fontsize = 10)
print("hola")