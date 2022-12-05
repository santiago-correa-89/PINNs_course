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

# Load Data Xdata refers to spacial position of point, Udata is the Velocity field and Pressure fields for the points. 
Xdata = np.load(r"src/data/VORT_DATA_VTU/Xdata.npy")
Udata = np.load(r"src/data/VORT_DATA_VTU/Udata.npy")
idxTest = myLoad(r'/results/idxTest1241913')
W = myLoad(r'/results/wResult1241913')
b = myLoad(r'/results/bResult1241913')
loss = myLoad(r'/results/lossResult1241913')

x = np.arange(-5, 15, 0.1)
y = np.arange(-5, 5, 0.05)
t = np.arange(0, 200, 1)

xNorm = myNormalize(x)
yNorm = myNormalize(y)
tNorm = myNormalize(t)

grid = arangeData(x, y, t)
phi, p = evalNNesc(grid, W, b)