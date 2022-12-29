import tensorflow as tf
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import os

from utilities import *
from postProcessing import *

def DNN(X, W, b): 
    lbx = -5.0
    ubx = 15
    xNorm = (2.0*(X[:, 0:1] - lbx)/(ubx - lbx) - 1.0)
    lby = -5
    uby = 5
    yNorm = (2.0*(X[:, 1:2] - lby)/(uby - lby) - 1.0)
    lbt = 0.0
    ubt = 2.0
    tNorm = (2.0*(X[:, 2:3] - lbt)/(ubt - lbt) - 1.0)

    A = tf.concat([xNorm, yNorm, tNorm],1)
    L = len(W)
    for i in range(L-1):
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y

def net_u(x, y, t, w, b):
    output  = DNN(tf.concat([x, y, t], 1), w, b)
    return output

folder = r'src/results/test'

date = str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)

W = myLoad(folder + '/wResult')
b = myLoad(folder + '/bResult')
loss = myLoad(folder + '/lossResult')
lossF = myLoad(folder + '/lossFResult')
lossU = myLoad(folder + '/lossUResult')

#Set of evaluation points
x = np.arange(-5, 15.1, 0.1)
y = np.arange(-5, 5.05, 0.05)
t = np.arange(0, 2.01, 0.01)

grid = arangeData(x, y, t, T=None)
grid_tf = tf.convert_to_tensor(grid, dtype=tf.float32)

Estimation = net_u(grid_tf[:, 0:1], grid_tf[:, 1:2], grid_tf[:, 2:3], W, b)

Processing(x, y, t, Estimation, loss, lossF, lossU, folder, date)