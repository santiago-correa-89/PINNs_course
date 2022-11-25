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

import tensorflow as tf
import numpy as np
import pandas as pd
import time

np.random.seed(seed=1234)
tf.random.set_seed(1234)
tf.config.experimental.enable_tensor_float_32_execution(False)

# Initalization of Network
def hyper_initial(size):
    in_dim = size[0]
    out_dim = size[1]
    std = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal(shape=size, stddev = std))

# Neural Network 
def DNN(X, W, b):
    A = X
    L = len(W)
    for i in range(L-1):
        A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y

def train_vars(W, b):
    return W + b

def net_u(x, y, t, w, b):
    output = DNN(tf.concat([x, y, t],1), w, b)
    return output

@tf.function
def net_f(x, y, t, W, b, I_Re):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, y, t])
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            
            with tf.GradientTape(persistent=True) as tape3:
                tape3.watch([x, y, t])
                output = net_u(x, y, t, W, b)
                psi = output[:,0:1]
                p = output[:,1:2]
       
            u = tape3.gradient(psi, y)
            v = -tape3.gradient(psi, x)
    
        u_x = tape2.gradient(u, x)
        u_y = tape2.gradient(u, y)
        u_t = tape2.gradient(u, t)
        v_x = tape2.gradient(v, x)
        v_y = tape2.gradient(v, y)
        v_t = tape2.gradient(v, t)
        p_x = tape2.gradient(p, x)
        p_y = tape2.gradient(p, y)    
    
    u_xx = tape2.gradient(u_x, x)
    u_yy = tape2.gradient(u_y, y)
    v_xx = tape2.gradient(v_x, x)
    v_yy = tape2.gradient(v_y, y)
    
    del tape1
    
    fx = u_t + (u*u_x + v*u_y) + p_x - I_Re*(u_xx + u_yy)
    fy = v_t + (u*v_x + v*v_y) + p_y - I_Re*(v_xx + v_yy)
    
    return fx, fy, u, v, p


#@tf.function(jit_compile=True)
@tf.function
def train_step(W, b, X_d_train_tf, uvp_train_tf, X_f_train_tf, opt, I_Re):
    # Select data for training
    x_d = X_d_train_tf[:, 0:1]
    y_d = X_d_train_tf[:, 1:2]
    t_d = X_d_train_tf[:, 2:3]
    
    x_f = X_f_train_tf[:, 0:1]
    y_f = X_f_train_tf[:, 1:2]
    t_f = X_f_train_tf[:, 2:3]
    
    with tf.GradientTape() as tape:
        tape.watch([W, b])
        output = net_u(x_d, y_d, t_d, W, b)
        
        with tf.GradientTape() as tape1:
            tape1.watch([x_d, y_d, t_d])
            output = net_u(x_d, y_d, t_d, W, b)
            psi = output[:, 0:1]
            p = output[:, 1:2]
       
        u = tape1.gradient(psi, y_d)
        v = -tape1.gradient(psi, x_d)         
        
        fx, fy, _, _, _ = net_f(x_f,y_f, t_f, W, b, I_Re)
        loss =  tf.reduce_mean(tf.square(u - uvp_train_tf[:,0:1])) 
        + tf.reduce_mean(tf.square(v - uvp_train_tf[:,1:2])) 
        + tf.reduce_mean(tf.square(p - uvp_train_tf[:,2:3]))
        + tf.reduce_mean(tf.square( fx ))
        + tf.reduce_mean(tf.square( fy ))
        
    grads = tape.gradient(loss, train_vars(W,b))
    opt.apply_gradients(zip(grads, train_vars(W,b)))
    return loss

# Defining variables
D = 1
nu = 0.01
Uinf = 1
I_Re = nu/(Uinf*D)   
noise = 0.0        
N_u = 100
N_f = 10000
Niter = 5000

layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
L = len(layers)
W = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)] 

Udata = np.load('/src/code/data/VORT_DATA_VTU/Udata.npy')
Xdata = np.load('/src/code/data/VORT_DATA_VTU/Xdata.npy')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
y = data['x'].flatten()[:,None]
u = data['x'].flatten()[:,None]
v = data['x'].flatten()[:,None]
p = data['x'].flatten()[:,None]

Exact = np.real(data['usol']).T
X, T = np.meshgrid(x,t)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              
# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)    
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
uu2 = Exact[:,0:1]
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
uu3 = Exact[:,-1:]

X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = lb + (ub-lb)*lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)

X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]

X_u_train_tf = tf.convert_to_tensor(X_u_train, dtype=tf.float32)
u_train_tf =   tf.convert_to_tensor(u_train, dtype=tf.float32)
X_f_train_tf = tf.convert_to_tensor(X_f_train, dtype=tf.float32)

lr = 1e-3
optimizer = tf.optimizers.Adam(learning_rate=lr)

start_time = time.time()
n=0
loss = []
while n <= Nmax:
    loss_= train_step(W, b, X_u_train_tf, u_train_tf, X_f_train_tf, optimizer, nu)
    loss.append(loss_)    
    print(f"Iteration is: {n} and loss is: {loss_}")
    n+=1

elapsed = time.time() - start_time                
print('Training time: %.4f' % (elapsed))

