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

def evalNNesc(x, y, t, W, b):
    x_tf = tf.convert_to_tensor(np.c_[x], dtype=tf.float32)
    y_tf = tf.convert_to_tensor(np.c_[y], dtype=tf.float32)
    t_tf = tf.convert_to_tensor(np.c_[t], dtype=tf.float32)
    
    predict = net_u(x_tf, y_tf, t_tf, W, b).numpy()
    
    return predict[0][0], predict[0][1]

W = myLoad('W_result')
b = myLoad('b_result')
loss = myLoad('loss_result')

x = 2.0
y = 2.0
t = 2.0

phi, p = evalNNesc(x, y, t, W, b)

allx = np.arange(-5,15,0.5)
ally = np.arange(-5,5, 0.5)
allt = np.arange(0, 200, 3)

allphi = np.zeros((len(allx),len(ally),len(allt)))
allp = np.zeros((len(allx),len(ally),len(allt)))

for x in allx:
    for y in ally:
        for t in allt:
            phi, p = evalNNesc(x, y, t, W, b)
            allphi[np.where(allx==x)[0][0],np.where(ally==y)[0][0],np.where(allt==t)[0][0]] = phi
            allp[np.where(allx==x)[0][0],np.where(ally==y)[0][0],np.where(allt==t)[0][0]] = p
            #print(t)
    print(x)

Xmesh, Ymesh = np.meshgrid(allx, ally)            
fig, ax = plt.subplots()
hm = ax.imshow(allp[:,:,100].T, extent=[Xmesh.min(), Xmesh.max(), Ymesh.max(), Ymesh.min()]) 


lossnp = tf.concat(loss,0).numpy()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lossnp)
ax.set_xlabel('$n iter$')
ax.set_ylabel('$loss$')    
ax.set_title('Loss evolution', fontsize = 10)
print("hola")

#phi_predict = []
#p_predict = []
#u_predict = []
#v_predict = []


# x_predict = np.vstack(np.arange(-5,15, 0.1))
# y_predict = np.vstack(np.arange(-5,5, 0.05))
# t_predict = np.vstack(np.arange(0, 200, 1))

# x_predict = tf.convert_to_tensor(x_predict, dtype=tf.float32)
# y_predict = tf.convert_to_tensor(y_predict, dtype=tf.float32)
# t_predict = tf.convert_to_tensor(t_predict, dtype=tf.float32)

# predict = net_u(x_predict, y_predict, t_predict, W_, b_)
#      p_predict.append(predict[:, 0:1])
#         phi_predict = predict[:, 1:2]
    
#     u_ = tape6.gradient(phi_predict, y_predict)
#     v_ = -tape6.gradient(phi_predict, x_predict)
    
#     p_predict = p_predict.numpy().flatten()
# heatmap, xi, yi, zi = nonuniform_imshow( x_predict, y_predict, p_predict )

# fig, ax = newfig(1.0, 1.1)
# ax.axis('off')

# gs0 = gridspec.GridSpec(1, 2)
# gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
# ax = plt.subplot(gs0[:, :])

# img = plt.colorbar(heatmap)
# plt.title('Field P (Pa)')
# plt.xlabel('coord x')
# plt.ylabel('coord y')
# plt.savefig(r"src/data/fig/Presion/P_predict" + str(j) + ".png")
# plt.ion()
# plt.show()
# plt.pause(1)