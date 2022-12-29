import tensorflow as tf
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

import os

from utilities import *

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

def train_vars(W, b):
    return W + b

def net_u(x, y, t, w, b):
    output  = DNN(tf.concat([x, y, t], 1), w, b)
    return output

def net_f(x, y, t, W, b, var):
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
    
    u_xx = tape1.gradient(u_x, x)
    u_yy = tape1.gradient(u_y, y)
    v_xx = tape1.gradient(v_x, x)
    v_yy = tape1.gradient(v_y, y)
    
    del tape1
    
    fx = u_t + (u*u_x + v*u_y) + p_x - var*(u_xx + u_yy)
    fy = v_t + (u*v_x + v*v_y) + p_y - var*(v_xx + v_yy)
    
    return fx, fy, u, v, p

def train_step(W, b, X_d_train_tf, uvp_train_tf, X_f_train_tf, opt, var):
    # Select data for training
    alp = 0.6
    x_d = X_d_train_tf[:, 0:1]
    y_d = X_d_train_tf[:, 1:2]
    t_d = X_d_train_tf[:, 2:3]
    
    x_f = X_f_train_tf[:, 0:1]
    y_f = X_f_train_tf[:, 1:2]
    t_f = X_f_train_tf[:, 2:3]
    
    with tf.GradientTape(persistent=True) as tape4:
        tape4.watch([W, b, var])
        
        with tf.GradientTape(persistent=True) as tape5:
            tape5.watch([x_d, y_d, t_d])
            output = net_u(x_d, y_d, t_d, W, b)
            psi = output[:, 0:1]
            p = output[:, 1:2]
       
        u = tape5.gradient(psi, y_d)
        v = -tape5.gradient(psi, x_d)
        
        del tape5    
        
        fx, fy, _, _, _ = net_f(x_f, y_f, t_f, W, b, var)
        
        lossD =  tf.reduce_mean(tf.square(u - uvp_train_tf[:,0:1])) \
        + tf.reduce_mean(tf.square(v - uvp_train_tf[:,1:2])) \
        + tf.reduce_mean(tf.square(p - uvp_train_tf[:,2:3]))
                    
        lossF = tf.reduce_mean(tf.square( fx )) \
        + tf.reduce_mean(tf.square( fy ))
        
        loss = lossD*alp+ lossF*(1-alp)

    grad1 = tape4.gradient(loss, var)
    opt.apply_gradients(zip([grad1], [var]))
            
    grads = tape4.gradient(loss, train_vars(W,b))
    opt.apply_gradients(zip(grads, train_vars(W,b)))

    del tape4
    return loss, lossD, lossF, W, b, var

def predict(Xtest, W, b):
    x = Xtest[:, 0:1]
    y = Xtest[:, 1:2]
    t = Xtest[:, 2:3]
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y, t])
        output = net_u(x, y, t, W, b)
        psi = output[:,0:1]
        p = output[:,1:2]
       
    u = tape.gradient(psi, y)
    v = -tape.gradient(psi, x)
    return u, v, p

if __name__ == "__main__": 
# Defining variables
    # Defining Parameters
    # D = 1
    # nu = 0.01
    # Uinf = 1
    # I_Re = nu/(Uinf*D) 
    var = tf.Variable([1.0], dtype=tf.float32)
    
    noise = 0.0        
    
    # Defininig training parameters
    Ntest = 100
    Ndata = 40
    Nfis = 10000 
    Niter = 4000
    T=201

    # Defining Neural Network
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    L = len(layers)
    W = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
    b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)] 

    # Load Data Xdata refers to spacial position of point, Udata is the Velocity field and Pressure fields for the points. 
    Xdata = np.load(r"src/data/VORT_DATA_VTU/Xdata.npy")
    Udata = np.load(r"src/data/VORT_DATA_VTU/Udata.npy")

    # Select a number of point to test the NN
    idxTest = select_idx(Xdata, Ntest, criterion='uni')
    Xtest, Utest = conform_data(Xdata, Udata, idxTest)
    Xtest = Xtest[(Utest[:,0]>0.01)*(Utest[:,1]>0.01)*(Utest[:,2]>0.01)] # Filter to avoide data (u, v, p) near zero
    Utest = Utest[(Utest[:,0]>0.01)*(Utest[:,1]>0.01)*(Utest[:,2]>0.01)] # Filter to avoide data (u, v, p) near zero
    Xtest_tf = tf.convert_to_tensor(Xtest, dtype=tf.float32)
    #Utest_tf = tf.convert_to_tensor(Utest, dtype=tf.float32)
    
    # Remove index used for test
    Xdata = np.delete(Xdata, idxTest, axis=0)
    Udata = np.delete(Udata, idxTest, axis=0)
    
    idxTrain = select_idx(Xdata, Ndata, criterion='uni')
    X_d_train, U_d_train = conform_data(Xdata, Udata, idxTrain)
    
    ptsF = np.random.uniform([-5, -5], [15, 5], size=(Nfis, 2))  #interior fis points w no data
    X_f_train = np.c_[ptsF, 0.01*np.random.randint(T, size=Nfis) ]
    X_f_train = np.vstack([X_f_train, circle_points(Nfis)]) #border fis points w no data
    X_f_train = np.vstack([X_f_train, X_d_train]) #eval fis in data points
 
    X_d_train_tf = tf.convert_to_tensor(X_d_train, dtype=tf.float32)
    U_d_train_tf = tf.convert_to_tensor(U_d_train, dtype=tf.float32)
    X_f_train_tf = tf.convert_to_tensor(X_f_train, dtype=tf.float32)

    lr = 1e-3
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    start_time = time.time()
    n=0
    loss = []
    lossD = []
    lossF = []

    while n <= Niter:
        loss_, lossD_, lossF_, W, b, var = train_step(W, b, X_d_train_tf, U_d_train_tf, X_f_train_tf, optimizer, var )
        loss.append(loss_.numpy())
        lossD.append(lossD_.numpy())
        lossF.append(lossF_.numpy())   
        if(n %10 == 0):   
            print(f"Iteration is: {n} and loss is: {loss_}")
            print(f"var = {var.numpy()[0]}")
        n+=1

    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

    # Save parameteres for postprocessing
    date = str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
    mySave('src/results/wResult' + date, W)
    mySave('src/results/bResult' + date, b)
    mySave('src/results/lossResult' + date, loss)
    mySave('src/results/lossDResult' + date, lossD)
    mySave('src/results/lossFResult' + date, lossF)
    
    # Prediction of NN for test points
    uPredict, vPredict, pPredict = predict(Xtest_tf, W, b)
    
   # Error estimation for prediction of NN
    errU = ((Utest[:, 0:1] - uPredict.numpy())/Utest[:, 0:1])*100
    errV = ((Utest[:, 1:2] - vPredict.numpy())/Utest[:, 1:2])*100
    errP = ((Utest[:, 2:3] - pPredict.numpy())/Utest[:, 2:3])*100
    
    np.save(r"results/error.npy", [errU, errV, errP])
    
    meanErrU = np.mean(np.abs(errU)) #np.linalg.norm(Utest[:, 0:1] - uPredict.numpy(), 2)/np.linalg.norm(Utest[:, 0:1], 2)
    meanErrV = np.mean(np.abs(errV))
    meanErrP = np.mean(np.abs(errP))
    
    np.save(r"results/Xtest.npy", Xtest)
    
    print("Percentual errors are {:.2f} in u, {:.2f} in v and {:.2f} in p.".format(errU, errV, errP))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lossF, 'r--', lossD, 'bs', loss, 'g^')
    ax.set_xlabel('$n iter$')
    ax.set_ylabel('Loss')
    plt.yscale('log')
    ax.set_title('Loss evolution', fontsize = 10)
    fig.show()
    
    print('Hello World')