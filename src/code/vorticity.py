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

def net_f(x, y, t, W, b, I_Re):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, y, t])
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            
            with tf.GradientTape(persistent=True) as tape3:
                tape3.watch([x, y, t])
                
                with tf.GradientTape(persistent=True) as tape4:
                    tape4.watch([x, y, t])
                    output = net_u(x, y, t, W, b)
                    psi = output[:,0:1]
                    p = output[:,1:2]
       
                u = tape4.gradient(psi, y)
                v = -tape4.gradient(psi, x)
    
            u_x = tape3.gradient(u, x)
            u_y = tape3.gradient(u, y)
            u_t = tape3.gradient(u, t)
            v_x = tape3.gradient(v, x)
            v_y = tape3.gradient(v, y)
            v_t = tape3.gradient(v, t)
            p_x = tape3.gradient(p, x)
            p_y = tape3.gradient(p, y)
            w = v_x - u_y
        
        w_x = tape2.gradient(w, x)
        w_y = tape2.gradient(w, y)
        w_t = tape2.gradient(w, t)    
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
    
    w_xx = tape1.gradient(w_x, x)
    w_yy = tape1.gradient(w_y, y)
        
    del tape1
    
    fx = u_t + (u*u_x + v*u_y) + p_x - I_Re*(u_xx + u_yy)
    fy = v_t + (u*v_x + v*v_y) + p_y - I_Re*(v_xx + v_yy)
    fw = w_t + u*w_x + v*w_y - I_Re*(w_xx + w_yy) 
    
    return fx, fy, fw, u, v, p, w

def train_step(W, b, X_d_train_tf, uvpw_train_tf, X_f_train_tf, opt, I_Re):
    # Select data for training
    x_d = X_d_train_tf[:, 0:1]
    y_d = X_d_train_tf[:, 1:2]
    t_d = X_d_train_tf[:, 2:3]
    
    x_f = X_f_train_tf[:, 0:1]
    y_f = X_f_train_tf[:, 1:2]
    t_f = X_f_train_tf[:, 2:3]
    
    with tf.GradientTape(persistent=True) as tape5:
        alp = 0.7
        tape5.watch([W, b])
        
        with tf.GradientTape(persistent=True) as tape6:
            tape6.watch([x_d, y_d, t_d])
            with tf.GradientTape(persistent=True) as tape7:
                tape7.watch([x_d, y_d, t_d])
                output = net_u(x_d, y_d, t_d, W, b)
                psi = output[:, 0:1]
                p = output[:, 1:2]
       
            u = tape7.gradient(psi, y_d)
            v = -tape7.gradient(psi, x_d)
        
        v_x = tape6.gradient(v, x_d)
        u_y = tape6.gradient(u, y_d)
        w = v_x - u_y
        
        del tape6   
        
        fx, fy, fw, _, _, _, _ = net_f(x_f, y_f, t_f, W, b, I_Re)
        lossD =  tf.reduce_mean(tf.square(u - uvpw_train_tf[:,0:1])) \
        + tf.reduce_mean(tf.square(v - uvpw_train_tf[:,1:2])) \
        + tf.reduce_mean(tf.square(p - uvpw_train_tf[:,2:3])) \
        + tf.reduce_mean(tf.square(w - uvpw_train_tf[:,3:4]))
                    
        lossF = tf.reduce_mean(tf.square( fx )) \
        + tf.reduce_mean(tf.square( fy )) \
        + tf.reduce_mean(tf.square( fw ))
        
        loss = alp*lossD + (1 - alp)*lossF
        
    grads = tape5.gradient(loss, train_vars(W,b))
    opt.apply_gradients(zip(grads, train_vars(W,b)))
    del tape5
    return loss, lossD, lossF, W, b

def predict(Xtest, W, b):
    x = Xtest[:, 0:1]
    y = Xtest[:, 1:2]
    t = Xtest[:, 2:3]
    
    with tf.GradientTape(persistent=True) as tape8:
        tape8.watch([x, y, t])
        with tf.GradientTape(persistent=True) as tape9:
            tape9.watch([x, y, t])
            output = net_u(x, y, t, W, b)
            psi = output[:,0:1]
            p = output[:,1:2]
       
        u = tape9.gradient(psi, y)
        v = -tape9.gradient(psi, x)
    
    v_x = tape8.gradient(v, x)
    u_y = tape8.gradient(u, y)
    w = v_x - u_y
    
    return u, v, p, w

if __name__ == "__main__": 
# Defining variables
    D = 1
    nu = 0.01
    Uinf = 1
    I_Re = nu/(Uinf*D)   
    noise = 0.0        
    Ntest = 100
    Ndata = 40
    Nfis = 5000 
    Niter = 4000
    T=201

    # Defining Neural Network
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    L = len(layers)
    W = [hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
    b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)] 

    # Load Data Xdata refers to spacial position of point, Udata is the Velocity field and Pressure fields for the points. 
    Xdata = np.load(r"src/data/vorticityTest/Xdata.npy")
    Udata = np.load(r"src/data/vorticityTest/Udata.npy")

    # Select a number of point to test the NN
    idxTest = select_idx(Xdata, Ntest, criterion='uni')
    Xtest, Utest = conform_data(Xdata, Udata, idxTest)
    Xtest = Xtest[(Utest[:,0]>0.01)*(Utest[:,1]>0.01)*(Utest[:,2]>0.01)*(Utest[:,3]>0.01)] # Filter to avoide data (u, v, p) near zero
    Utest = Utest[(Utest[:,0]>0.01)*(Utest[:,1]>0.01)*(Utest[:,2]>0.01)*(Utest[:,3]>0.01)] # Filter to avoide data (u, v, p) near zero
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
        loss_, lossD_, lossF_, W, b = train_step(W, b, X_d_train_tf, U_d_train_tf, X_f_train_tf, optimizer, I_Re)
        loss.append(loss_)
        lossD.append(lossD_)
        lossF.append(lossF_)   
        if(n %1 == 0):   
            print(f"Iteration is: {n} and loss is: {loss_}")
        n+=1

    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

    # Save parameteres for postprocessing
    date = str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
    mySave('src/results/vorticityTest/wResult' + date, W)
    mySave('src/results/vorticityTest/bResult' + date, b)
    mySave('src/results/vorticityTest/lossResult' + date, loss)
    mySave('src/results/vorticityTest/lossDResult' + date, lossD)
    mySave('src/results/vorticityTest/lossFResult' + date, lossF)
    
    # Prediction of NN for test points
    uPredict, vPredict, pPredict, wPredict = predict(Xtest_tf, W, b)
    
    # Error estimation for prediction of NN
    errU = (Utest[:, 0:1] - uPredict.numpy())/Utest[:, 0:1]
    errV = (Utest[:, 1:2] - vPredict.numpy())/Utest[:, 1:2]
    errP = (Utest[:, 2:3] - pPredict.numpy())/Utest[:, 2:3]
    errW = (Utest[:, 3:4] - wPredict.numpy())/Utest[:, 3:4]

    
    np.save(r"src/results/vorticityTest/error.npy", [errU, errV, errP, errW])
    
    meanErrU = np.mean(np.abs(errU))*100 #np.linalg.norm(Utest[:, 0:1] - uPredict.numpy(), 2)/np.linalg.norm(Utest[:, 0:1], 2)
    meanErrV = np.mean(np.abs(errV))*100
    meanErrP = np.mean(np.abs(errP))*100
    meanErrW = np.mean(np.abs(errW))*100
    
    np.save(r"src/results/vorticityTest/Xtest.npy", Xtest)
    
    print("Percentual errors are {:.2f} in u, {:.2f} in v, {:.2f} in p and {:.2f} in w.".format(meanErrU, meanErrV, meanErrP, meanErrW))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lossF, 'r--', lossD, 'bs', loss, 'g^')
    ax.set_xlabel('$n iter$')
    ax.set_ylabel('Loss')
    plt.yscale('log')
    ax.set_title('Loss evolution 40 Data Samples Normalized', fontsize = 10)
    fig.show()
    
print('hello world')