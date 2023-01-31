import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from PIL import Image 

from scipy import spatial
from pyDOE import lhs

import pickle
import os

from utilities import *
#from flatWeightsLBFGS import *
from postProcessing import *

np.random.seed(seed=1234)
tf.random.set_seed(1234)
tf.config.experimental.enable_tensor_float_32_execution(False)

class NavierStoke:
    #Initialize the class
    
    def __init__(self, X_d, U_d, X_f, X_b, U_b, X_i, U_i, Xtest, Utest, xGrid, yGrid, tGrid, layers, lr, Re, folder, nIterAdam):
        
        self.nIterAdam = nIterAdam
        self.N_f = X_f.shape[0]
        self.N_b = X_b.shape[0]
        self.N_i = X_i.shape[0]
        self.Re = Re
        self.lb = X_d.min(0)
        self.ub = X_d.max(0)
        
        #Training data Points
        self.xTrain_tf = tf.convert_to_tensor(X_d[:, 0:1], dtype=tf.float32)
        self.yTrain_tf = tf.convert_to_tensor(X_d[:, 1:2], dtype=tf.float32)
        self.tTrain_tf = tf.convert_to_tensor(X_d[:, 2:3], dtype=tf.float32)
        self.XdataTrain_tf = tf.convert_to_tensor(X_d, dtype=tf.float32)
        
        #Training data Variables
        self.uTrain = U_d[:, 0:1]
        self.vTrain = U_d[:, 1:2]
        self.pTrain = U_d[:, 2:3]
        self.UdataTrain = tf.convert_to_tensor(U_d, dtype=tf.float32)
        
        #Physics points for training
        self.xPhisic_tf = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32)
        self.yPhisic_tf = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32)
        self.tPhisic_tf = tf.convert_to_tensor(X_f[:, 2:3], dtype=tf.float32)
        self.Xphisics_tf = tf.convert_to_tensor(X_f, dtype=tf.float32)

        #Boundary points for training
        self.xBound_tf = tf.convert_to_tensor(X_b[:, 0:1], dtype=tf.float32)
        self.yBound_tf = tf.convert_to_tensor(X_b[:, 1:2], dtype=tf.float32)
        self.tBound_tf = tf.convert_to_tensor(X_b[:, 2:3], dtype=tf.float32)
        self.uBound = U_b[:, 0:1]
        self.vBound = U_b[:, 1:2]
        self.pBound = U_b[:, 2:3]
        
        #Initial points for training
        self.xInit_tf = tf.convert_to_tensor(X_i[:, 0:1], dtype=tf.float32)
        self.yInit_tf = tf.convert_to_tensor(X_i[:, 1:2], dtype=tf.float32)
        self.tInit_tf = tf.convert_to_tensor(X_i[:, 2:3], dtype=tf.float32)
        self.Xinit_tf = tf.convert_to_tensor(X_i, dtype=tf.float32)
        self.uInit = U_i[:, 0:1]
        self.vInit = U_i[:, 1:2]
        self.pInit = U_i[:, 2:3]

        #Test points
        self.xTest_tf = tf.convert_to_tensor(Xtest[:, 0:1], dtype=tf.float32)
        self.yTest_tf = tf.convert_to_tensor(Xtest[:, 1:2], dtype=tf.float32)
        self.tTest_tf = tf.convert_to_tensor(Xtest[:, 2:3], dtype=tf.float32)
        self.XTest_tf = tf.convert_to_tensor(Xtest, dtype=tf.float32)
        
        #Test data for training
        self.uTest = Utest[:, 0:1]
        self.vTest = Utest[:, 1:2]
        self.pTest = Utest[:, 2:3]

        #Set of Estimation point to evaluete prediction
        self.grid = arangeData(xGrid, yGrid, tGrid, T=None)
        self.grid_tf = tf.convert_to_tensor(self.grid, dtype=tf.float32)

        #NN structure
        self.layers = layers
        self.W, self.b = self.InitializeNN(self.layers)
        self.lambda_f, self.lambda_b, self.lambda_i = self.Initialize_mults(self.N_f, self.N_b, self.N_i)
        
        #NN first optimizer
        self.optimizer_adam_nn = tf.optimizers.Adam(learning_rate=lr)
        self.optimizer_adam_f = tf.optimizers.Adam(learning_rate=lr)
        self.optimizer_adam_b = tf.optimizers.Adam(learning_rate=lr)
        self.optimizer_adam_i = tf.optimizers.Adam(learning_rate=lr)

        start_time = time.time()
        n=0
        self.loss = []
        self.lossU = []
        self.lossF = []
        self.lossB = []
        self.lossI = []

        #Optimization process with Adam
        while n <= self.nIterAdam:
            loss_, lossU_, lossF_, lossB_, lossI_, self.W, self.b, self.lambda_f, self.lambda_b, self.lambda_i = self.train_step(self.W, self.b, self.lambda_f, self.lambda_b, self.lambda_i)
            self.loss.append(loss_)
            self.lossU.append(lossU_)
            self.lossF.append(lossF_)
            self.lossB.append(lossB_)
            self.lossI.append(lossI_)
        
            if(n %100 == 0):   
                print(f"Iteration is: {n} and loss is: {loss_}")
            n+=1
        
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        
        # Save parameteres for postprocessing
        self.date = str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
    
        mySave(folder + '/wResult', self.W)
        mySave(folder + '/bResult', self.b)
        mySave(folder + '/lambda_fResult', self.lambda_f)
        mySave(folder + '/lambda_bResult', self.lambda_b)
        mySave(folder + '/lambda_iResult', self.lambda_i)
        mySave(folder + '/lossResult', self.loss)
        mySave(folder + '/lossUResult', self.lossU)
        mySave(folder + '/lossFResult', self.lossF)
        mySave(folder + '/lossBResult', self.lossB)
        mySave(folder + '/lossIResult', self.lossI)
        
        # Prediction of NN for test points
        self.uPredict, self.vPredict, self.pPredict = self.predict(self.xTest_tf, self.yTest_tf, self.tTest_tf, self.W, self.b)
        
        # Error estimation for prediction of NN
        self.errU = (self.uTest - self.uPredict.numpy())
        self.errV = (self.vTest - self.vPredict.numpy())
        self.errP = (self.pTest - self.pPredict.numpy())
    
        np.save(folder + '/error.npy', [self.errU, self.errV, self.errP])
    
        self.meanErrU = (np.linalg.norm(self.errU)/np.linalg.norm(self.uTest + 1e-8))*100 
        self.meanErrV = (np.linalg.norm(self.errV)/np.linalg.norm(self.vTest + 1e-8))*100
        self.meanErrP = (np.linalg.norm(self.errP)/np.linalg.norm(self.pTest + 1e-8))*100

        np.save(folder + '/meanError.npy', [self.meanErrU, self.meanErrV, self.meanErrP])

        np.save(folder + '/Xtest.npy', Xtest)
    
        print("Percentual errors are {:.2f} in u, {:.2f} in v and {:.2f} in p.".format(self.meanErrU, self.meanErrV, self.meanErrP))

        #Post processing using a grid to predict the flow variation   
        self.Estimation = self.net_u(self.grid_tf[:, 0:1], self.grid_tf[:, 1:2], self.grid_tf[:, 2:3], self.W, self.b)
        
        Processing(xGrid, yGrid, tGrid, self.Estimation, self.loss, self.lossF, self.lossU, folder, self.date)
    
    # Initalization of Network
    def hyper_initial(self, size):
        in_dim = size[0]
        out_dim = size[1]
        std = np.sqrt(2.0/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal(shape=size, stddev = std))
    
    def InitializeNN(self, layers):
        L = len(layers)
        W = [self.hyper_initial([layers[l-1], layers[l]]) for l in range(1, L)] 
        b = [tf.Variable(tf.zeros([1, layers[l]])) for l in range(1, L)]
        return W, b

    def Initialize_mults(self, N_f, N_b, N_i):
        lambda_f = tf.Variable(tf.ones([N_f, 1], dtype=tf.float32))
        lambda_b = tf.Variable(tf.ones([N_b, 1], dtype=tf.float32))
        lambda_i = tf.Variable(tf.ones([N_i, 1], dtype=tf.float32))
        return lambda_f, lambda_b, lambda_i
    # Neural Network 
    def DNN(self, X, W, b):
        A = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0   
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        return Y

    def train_vars_nn(self, W, b):
        return W + b
    
    def train_vars_sa(self, lambda_f, lambda_b, lambda_i):
        return lambda_f+lambda_b+lambda_i

    def train_vars_all(self, W, b, lambda_f, lambda_b, lambda_i):
        return W + b + lambda_f+lambda_b+lambda_i

    def net_u(self, x, y, t, w, b):
        output  = self.DNN(tf.concat([x, y, t], 1), w, b)
        return output

    def net_f(self, x, y, t, W, b):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, y, t])
        
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([x, y, t])
            
                with tf.GradientTape(persistent=True) as tape3:
                    tape3.watch([x, y, t])
                    output = self.net_u(x, y, t, W, b)
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
    
        fx = u_t + (u*u_x + v*u_y) + p_x - self.Re*(u_xx + u_yy)
        fy = v_t + (u*v_x + v*v_y) + p_y - self.Re*(v_xx + v_yy)
        return fx, fy


    def train_step(self, W, b,lambda_f, lambda_b, lambda_i):
        #self.optimizer_adam.build([W, b, lambda_f, lambda_b, lambda_i])
        with tf.GradientTape(persistent=True) as tape4:
            tape4.watch([W, b, lambda_f, lambda_b, lambda_i])
        
            with tf.GradientTape(persistent=True) as tape5:
                tape5.watch([self.xTrain_tf, self.yTrain_tf, self.tTrain_tf])
                output = self.net_u(self.xTrain_tf, self.yTrain_tf, self.tTrain_tf, W, b)
                psi = output[:, 0:1]
                p = output[:, 1:2]
       
            u = tape5.gradient(psi, self.yTrain_tf)
            v = -tape5.gradient(psi, self.xTrain_tf)
            del tape5

            with tf.GradientTape(persistent=True) as tape6:
                tape6.watch([self.xBound_tf, self.yBound_tf, self.tBound_tf])
                output = self.net_u(self.xBound_tf, self.yBound_tf, self.tBound_tf, W, b)
                psi_B = output[:, 0:1]
                p_B = output[:, 1:2]
       
            u_B = tape6.gradient(psi_B, self.yBound_tf)
            v_B = -tape6.gradient(psi_B, self.xBound_tf)
            del tape6

            with tf.GradientTape(persistent=True) as tape7:
                tape7.watch([self.xInit_tf, self.yInit_tf, self.tInit_tf])
                output = self.net_u(self.xInit_tf, self.yInit_tf, self.tInit_tf, W, b)
                psi_I = output[:, 0:1]
                p_I = output[:, 1:2]
       
            u_I = tape7.gradient(psi_I, self.yInit_tf)
            v_I = -tape7.gradient(psi_I, self.xInit_tf)
            del tape7 
        
            fx, fy = self.net_f(self.xPhisic_tf, self.yPhisic_tf, self.tPhisic_tf, W, b)
            
            lossU = tf.reduce_mean(tf.square(u - self.uTrain)) \
            + tf.reduce_mean(tf.square(v - self.vTrain)) \
            + tf.reduce_mean(tf.square(p - self.pTrain))
                    
            lossF = tf.reduce_mean(tf.square( lambda_f*fx )) + tf.reduce_mean(tf.square( lambda_f*fy ))

            lossB = tf.reduce_mean(tf.square( lambda_b*(u_B - self.uBound) )) \
            + tf.reduce_mean(tf.square( lambda_b*(v_B - self.vBound) )) \
            + tf.reduce_mean(tf.square( lambda_b*(p_B - self.pBound )))

            lossI = tf.reduce_mean(tf.square( lambda_i*(u_I - self.uInit) )) \
            + tf.reduce_mean(tf.square( lambda_i*(v_I - self.vInit) )) \
            + tf.reduce_mean(tf.square( lambda_i*(p_I - self.pInit) ))

            loss = lossU + lossF + lossB + lossI
        
        grads = tape4.gradient(loss, self.train_vars_nn(W,b))
        grads_f = tape4.gradient(loss, lambda_f)
        grads_b = tape4.gradient(loss, lambda_b)
        grads_i = tape4.gradient(loss, lambda_i)
        self.optimizer_adam_nn.apply_gradients(zip(grads, self.train_vars_nn(W,b)))
        self.optimizer_adam_f.apply_gradients(zip([-grads_f], [lambda_f]))
        self.optimizer_adam_b.apply_gradients(zip([-grads_b], [lambda_b]))
        self.optimizer_adam_i.apply_gradients(zip([-grads_i], [lambda_i]))

        del tape4
        
        return loss, lossU, lossF, lossB, lossI, W, b, lambda_f, lambda_b, lambda_i
    
    def predict(self, xTest, yTest, tTest, W, b):
    
        with tf.GradientTape(persistent=True) as tape6:
            tape6.watch([xTest, yTest, tTest])
            output = self.net_u(xTest, yTest, tTest, W, b)
            psi = output[:,0:1]
            p = output[:,1:2]
       
        u = tape6.gradient(psi, yTest)
        v = -tape6.gradient(psi, xTest)
        return u, v, p
    

if __name__ == "__main__": 
# Defining variables
    D = 1 
    nu = 0.01
    Uinf = 1
    Re = nu/(Uinf*D)   
    noise = 0.0        
    Ntest = 200
    Ndata = 40
    Ninit = 1000
    Nfis = 15000
    Ncyl = 10000
    nIterAdam = 20#000
    
    T=201
    tInit = 15 # Initial time

    # Defining Neural Network
    layers = [3]+6*[64]+[2]

    # Load Data Xdata refers to spacial position of point, Udata is the Velocity field and Pressure fields for the points. 
    Xdata = np.load(r"src/data/vorticityTest/Xdata.npy")
    Udata = np.load(r"src/data/vorticityTest/Udata.npy")[:,:,tInit:T]
    
    # Boundary Conditions
    _, upperBConditionIdx = spatial.KDTree(Xdata).query(Xdata[(Xdata[:,1] == 5)])
    upperBConditionIdx = np.random.choice(upperBConditionIdx, 20, replace=False)
    XupBC, UpBC = conform_data(Xdata, Udata, upperBConditionIdx, T=None)

    _, underBConditionIdx = spatial.KDTree(Xdata).query(Xdata[(Xdata[:,1] == -5)])
    underBConditionIdx = np.random.choice(underBConditionIdx, 20, replace=False)
    XunBC, UnBC = conform_data(Xdata, Udata, underBConditionIdx, T=None)
    
    _, inletBConditionIdx = spatial.KDTree(Xdata).query(Xdata[(Xdata[:,0] == -5)])
    inletBConditionIdx = np.random.choice(inletBConditionIdx, 20, replace=False)
    XinBC, InBC = conform_data(Xdata, Udata, inletBConditionIdx, T=None)

    # Boundary Conditions structure
    r = 0.5
    angle = np.pi*np.random.uniform(0, 2, Nfis)
    pts = np.array([r*np.cos(angle), r*np.sin(angle)]).T
    _, idx = spatial.KDTree(Xdata).query(pts) 
    idxCy =  np.unique(idx)
    XcyBC, CyBC = conform_data(Xdata, Udata, idxCy, T=None)

    idxBC = np.concatenate((underBConditionIdx, inletBConditionIdx, upperBConditionIdx, idxCy), axis=0, out=None)
    X_bc = np.concatenate((XunBC, XinBC, XupBC, XcyBC), axis=0, out=None)
    U_bc = np.concatenate((UnBC, InBC, UpBC, CyBC), axis=0, out=None) 

    # Initial Conditions
    idxIni = select_idx(Xdata, Ninit, criterion='uni')
    X_ini = Xdata[idxIni,:]
    X_ini = np.c_[X_ini, 0.01*np.random.randint(T-tInit, size=Ninit) ]
    U_ini = Udata[idxIni,:,0]

    # Select a number of point to test the NN
    idxTest = select_idx(Xdata, Ntest, criterion='uni')
    Xtest, Utest = conform_data(Xdata, Udata, idxTest)

    idxTrain = select_idx(Xdata, Ndata, criterion='lhs')
    XdataTrain, UdataTrain = conform_data(Xdata, Udata, idxTrain)
    #XdataTrain = np.concatenate((XunBC, XupBC, XinBC, XcyBC, XdataTrain))
    #UdataTrain = np.concatenate((UnBC, UpBC, InBC, CyBC, UdataTrain))
    
    ptsF = np.random.uniform([-5, -5], [15, 5], size=(Nfis, 2))  #interior fis points w no data
    X_f = np.c_[ptsF, 0.01*np.random.randint(T-tInit, size=Nfis) ]
    #X_f = np.vstack([X_f, XdataTrain]) #eval fis in data points

    lr = 1e-3
    
    folder = r'src/results/adaptive'

    #Set of evaluation points
    x = np.arange(-5, 15.1, 0.1)
    y = np.arange(-5, 5.05, 0.05)
    t = np.arange(0, (T-tInit)*0.01, 0.01)
    
    model = NavierStoke(XdataTrain, UdataTrain, X_f, X_bc, U_bc, X_ini, U_ini, Xtest, Utest, x, y, t, layers, lr, Re, folder, nIterAdam)