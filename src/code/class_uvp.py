import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from PIL import Image 

from scipy import spatial
from pyDOE import lhs

from utilities import *
from flatWeightsLBFGS import *
from postProcessing import *

np.random.seed(seed=1234)
tf.random.set_seed(1234)
tf.config.experimental.enable_tensor_float_32_execution(False)

class NavierStoke:
    #Initialize the class
    
    def __init__(self, X_d, U_d, X_f, X_b, U_b, X_i, U_i, Xtest, Utest, xGrid, yGrid, tGrid, layers, lr, Re, folder, nIterAdam, nIterLBFGS):
        
        N_f = X_f.shape[0]
        N_b = X_b.shape[0]
        N_i = X_i.shape[0]
        lb = X_d.min(0)
        ub = X_d.max(0)
       
        #Training data Points
        xTrain_tf = tf.convert_to_tensor(X_d[:, 0:1], dtype=tf.float32)
        yTrain_tf = tf.convert_to_tensor(X_d[:, 1:2], dtype=tf.float32)
        tTrain_tf = tf.convert_to_tensor(X_d[:, 2:3], dtype=tf.float32)
        
        #Training data Variables
        uTrain = U_d[:, 0:1]
        vTrain = U_d[:, 1:2]
        pTrain = U_d[:, 2:3]
        
        #Physics points for training
        xPhisic_tf = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32)
        yPhisic_tf = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32)
        tPhisic_tf = tf.convert_to_tensor(X_f[:, 2:3], dtype=tf.float32)

        #Boundary points for training
        xBound_tf = tf.convert_to_tensor(X_b[:, 0:1], dtype=tf.float32)
        yBound_tf = tf.convert_to_tensor(X_b[:, 1:2], dtype=tf.float32)
        tBound_tf = tf.convert_to_tensor(X_b[:, 2:3], dtype=tf.float32)
        uBound = U_b[:, 0:1]
        vBound = U_b[:, 1:2]
        pBound = U_b[:, 2:3]
        
        #Initial points for training
        xInit_tf = tf.convert_to_tensor(X_i[:, 0:1], dtype=tf.float32)
        yInit_tf = tf.convert_to_tensor(X_i[:, 1:2], dtype=tf.float32)
        tInit_tf = tf.convert_to_tensor(X_i[:, 2:3], dtype=tf.float32)
        uInit = U_i[:, 0:1]
        vInit = U_i[:, 1:2]
        pInit = U_i[:, 2:3]

        #Test points
        xTest_tf = tf.convert_to_tensor(Xtest[:, 0:1], dtype=tf.float32)
        yTest_tf = tf.convert_to_tensor(Xtest[:, 1:2], dtype=tf.float32)
        tTest_tf = tf.convert_to_tensor(Xtest[:, 2:3], dtype=tf.float32)
        
        #Test data for training
        uTest = Utest[:, 0:1]
        vTest = Utest[:, 1:2]
        pTest = Utest[:, 2:3]

        #Set of Estimation point to evaluete prediction
        grid = arangeData(xGrid, yGrid, tGrid, T=None)
        grid_tf = tf.convert_to_tensor(grid, dtype=tf.float32)

        #NN structure
        W, b = self.InitializeNN(layers)
        lambda_f, lambda_b, lambda_i = self.Initialize_mults(N_f, N_b, N_i)
        
        #NN first optimizer
        optimizer_adam_nn = tf.optimizers.Adam(learning_rate=lr)
        optimizer_adam_f  = tf.optimizers.Adam(learning_rate=lr)
        optimizer_adam_b  = tf.optimizers.Adam(learning_rate=lr)
        optimizer_adam_i  = tf.optimizers.Adam(learning_rate=lr)

        #NN structure
        W, b = self.InitializeNN(layers)

        # Initialize time and loss variables
        start_time = time.time()
        n=0
        self.loss  = []
        self.lossU = []
        self.lossF = []
        self.lossB = []
        self.lossI = []

        #Optimization process with Adam
        while n <= nIterAdam:
            loss_, lossU_, lossF_, lossB_, lossI_, W, b, lambda_f, lambda_b, lambda_i = self.train_step(xTrain_tf, yTrain_tf, tTrain_tf, xBound_tf, yBound_tf, tBound_tf, xInit_tf, yInit_tf, tInit_tf, xPhisic_tf, yPhisic_tf, tPhisic_tf, uTrain, vTrain, pTrain, uBound, vBound, pBound, uInit, vInit, pInit, W, b, lambda_f, lambda_b, lambda_i, optimizer_adam_nn, optimizer_adam_f, optimizer_adam_b, optimizer_adam_i, Re, lb, ub)
            self.loss.append(loss_)
            self.lossU.append(lossU_)
            self.lossF.append(lossF_)
            self.lossB.append(lossB_)
            self.lossI.append(lossI_)
        
            if(n %100 == 0):   
                print(f"Iteration is: {n} and loss is: {loss_}")
            n+=1
        
         # Create the LossAndFlatGradient instance
        loss_and_gradient = LossAndFlatGradient(self.train_vars_all(W, b, lambda_f, lambda_b, lambda_i), lambda: self.closureLoss(xTrain_tf, yTrain_tf, tTrain_tf, xBound_tf, yBound_tf, tBound_tf, xInit_tf, yInit_tf, tInit_tf, xPhisic_tf, yPhisic_tf, tPhisic_tf, uTrain, vTrain, pTrain, uBound, vBound, pBound, uInit, vInit, pInit, W, b, lambda_f, lambda_b, lambda_i, Re, lb, ub))    
        
        # Call lbfgs_minimize to optimize the neural network
        initial_weights = loss_and_gradient.to_flat_weights(self.train_vars_all(W, b, lambda_f, lambda_b, lambda_i))
        
        #Optimization process with LBFGS
        optimizer = tfp.optimizer.lbfgs_minimize(loss_and_gradient, initial_position=initial_weights,
        num_correction_pairs=100,
        tolerance=0,
        x_tolerance=0,
        f_relative_tolerance=0,
        max_iterations=nIterLBFGS,
        parallel_iterations=1,
        max_line_search_iterations=50)
        
        #Compute the optimized loss
        result = optimizer.objective_value
        self.loss.append(result)
        #output the optimized Weights and Bias
        loss_and_gradient.set_flat_weights(optimizer.position)
        
        elapsed = time.time() - start_time
        print('Training time: %.4f' % (elapsed))
        
        # Save parameteres for postprocessing
        self.date = str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
    
        mySave(folder + '/wResult', W)
        mySave(folder + '/bResult', b)
        mySave(folder + '/lambda_fResult', lambda_f)
        mySave(folder + '/lambda_bResult', lambda_b)
        mySave(folder + '/lambda_iResult', lambda_i)
        mySave(folder + '/lossResult', self.loss)
        mySave(folder + '/lossUResult', self.lossU)
        mySave(folder + '/lossFResult', self.lossF)
        mySave(folder + '/lossBResult', self.lossB)
        mySave(folder + '/lossIResult', self.lossI)
        
        # Prediction of NN for test points
        uPredict, vPredict, pPredict = self.predict(xTest_tf, yTest_tf, tTest_tf, W, b, lb, ub)
        
        # Mean Squared Error estimation for prediction of NN
        mseU = np.mean(np.square(uPredict.numpy() - uTest))
        mseP = np.mean(np.square(pPredict.numpy() - pTest))
        mseV = np.mean(np.square(vPredict.numpy() - vTest))

        maeU = np.mean(np.abs(uPredict.numpy() - uTest))
        maeP = np.mean(np.abs(pPredict.numpy() - pTest))
        maeV = np.mean(np.abs(vPredict.numpy() - vTest))

        countu = np.count_nonzero(np.abs(uPredict.numpy() - uTest) > 0.1)
        countv = np.count_nonzero(np.abs(vPredict.numpy() - vTest) > 0.1)
        countp = np.count_nonzero(np.abs(pPredict.numpy() - pTest) > 0.1)

        np.save(folder + '/Xtest.npy', Xtest)
    
        print("Mean Squared errors are {:.2f} in u, {:.2f} in v and {:.2f} in p.".format(mseU, mseV,mseP))
        print("Mean Abosulte errors are {:.2f} in u, {:.2f} in v and {:.2f} in p.".format(maeU, maeV,maeP))
        print("Count error > 10% {:.2f} in u, {:.2f} in v and {:.2f} in p.".format(countu, countv, countp))

        #Post processing using a grid to predict the flow variation   
        Estimation = self.net_u(grid_tf[:, 0:1], grid_tf[:, 1:2], grid_tf[:, 2:3], W, b, lb, ub)
        
        Processing(xGrid, yGrid, tGrid, Estimation, self.loss, self.lossF, self.lossU, folder, self.date)

        print('END')
    
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
        lambda_f = tf.Variable(10000*tf.ones([N_f, 1], dtype=tf.float32))
        lambda_b = tf.Variable(10000*tf.ones([N_b, 1], dtype=tf.float32))
        lambda_i = tf.Variable(10000*tf.ones([N_i, 1], dtype=tf.float32))
        return lambda_f, lambda_b, lambda_i
    
    # Neural Network 
    def DNN(self, X, W, b, lb, ub):
        A = 2.0*(X -lb)/(ub - lb) - 1.0   
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        return Y

    def train_vars_nn(self, W, b):
        return W + b
    
    def train_vars_sa(self, lambda_f, lambda_b, lambda_i):
        return lambda_f + lambda_b + lambda_i

    def train_vars_all(self, W, b, lambda_f, lambda_b, lambda_i):
        return W + b + [lambda_f] + [lambda_b] + [lambda_i]

    def net_u(self, x, y, t, w, b, lb, ub):
        output  = self.DNN(tf.concat([x, y, t], 1), w, b, lb, ub)
        return output

    def net_f(self, x, y, t, W, b, Re, lb, ub):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, y, t])
        
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([x, y, t])
        
                output = self.net_u(x, y, t, W, b, lb, ub)
                u = output[:,0:1]
                v = output[:,1:2]
                p = output[:,2:3]

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
    
        fx = u_t + (u*u_x + v*u_y) + p_x - Re*(u_xx + u_yy)
        fy = v_t + (u*v_x + v*v_y) + p_y - Re*(v_xx + v_yy)
        fm = u_y - v_x
        return fx, fy, fm


    def train_step(self, xTrain_tf, yTrain_tf, tTrain_tf, xBound_tf, yBound_tf, tBound_tf, xInit_tf, yInit_tf, tInit_tf, xPhisic_tf, yPhisic_tf, tPhisic_tf, uTrain, vTrain, pTrain, uBound, vBound, pBound, uInit, vInit, pInit, W, b, lambda_f, lambda_b, lambda_i, optimizer_adam_nn, optimizer_adam_f, optimizer_adam_b, optimizer_adam_i, Re, lb, ub):
        #self.optimizer_adam.build([W, b, lambda_f, lambda_b, lambda_i])
        with tf.GradientTape(persistent=True) as tape4:
            tape4.watch([W, b, lambda_f, lambda_b, lambda_i])

            u, v, p = self.trainSet(xTrain_tf, yTrain_tf, tTrain_tf, W, b, lb, ub)

            u_B, v_B, p_B = self.trainSet(xBound_tf, yBound_tf, tBound_tf, W, b, lb, ub)

            u_I, v_I, p_I = self.trainSet(xInit_tf, yInit_tf, tInit_tf, W, b, lb, ub)
        
            fx, fy, fm = self.net_f(xPhisic_tf, yPhisic_tf, tPhisic_tf, W, b, Re, lb, ub)
            
            lossU = tf.reduce_mean(tf.square(u - uTrain)) \
            + tf.reduce_mean(tf.square(v - vTrain)) \
            + tf.reduce_mean(tf.square(p - pTrain))
                    
            lossF = tf.reduce_mean(tf.square( lambda_f*fx )) + tf.reduce_mean(tf.square( lambda_f*fy)) + tf.reduce_mean(tf.square( fm ))

            lossB = tf.reduce_mean(tf.square( lambda_b*(u_B - uBound) )) \
            + tf.reduce_mean(tf.square( lambda_b*(v_B - vBound) )) \
            + tf.reduce_mean(tf.square( lambda_b*(p_B - pBound )))

            lossI = tf.reduce_mean(tf.square( lambda_i*(u_I - uInit) )) \
            + tf.reduce_mean(tf.square( lambda_i*(v_I - vInit) )) \
            + tf.reduce_mean(tf.square( lambda_i*(p_I - pInit) ))

            loss = lossU + lossF + lossB + lossI
        
        grads = tape4.gradient(loss, self.train_vars_nn(W,b))
        grads_f = tape4.gradient(loss, lambda_f)
        grads_b = tape4.gradient(loss, lambda_b)
        grads_i = tape4.gradient(loss, lambda_i)
        optimizer_adam_nn.apply_gradients(zip(grads, self.train_vars_nn(W,b)))
        optimizer_adam_f.apply_gradients(zip([-grads_f], [lambda_f]))
        optimizer_adam_b.apply_gradients(zip([-grads_b], [lambda_b]))
        optimizer_adam_i.apply_gradients(zip([-grads_i], [lambda_i]))

        del tape4
        
        return loss, lossU, lossF, lossB, lossI, W, b, lambda_f, lambda_b, lambda_i
    
    def predict(self, xTest, yTest, tTest, W, b, lb, ub):
    
        output = self.net_u(xTest, yTest, tTest, W, b, lb, ub)
        u = output[:,0:1]
        v = output[:,1:2]
        p = output[:,2:3]
        
        return u, v, p

    def trainSet(self, xData, yData, tData, W, b, lb, ub):
        
        output = self.net_u(xData, yData, tData, W, b, lb, ub)
        u = output[:,0:1]
        v = output[:,1:2]
        p = output[:,2:3]
       
        return u, v, p

    def closureLoss(self, xTrain_tf, yTrain_tf, tTrain_tf, xBound_tf, yBound_tf, tBound_tf, xInit_tf, yInit_tf, tInit_tf, xPhisic_tf, yPhisic_tf, tPhisic_tf, uTrain, vTrain, pTrain, uBound, vBound, pBound, uInit, vInit, pInit, W, b, lambda_f, lambda_b, lambda_i, Re, lb, ub):
        
        u, v, p = self.trainSet(xTrain_tf, yTrain_tf, tTrain_tf, W, b, lb, ub)

        u_B, v_B, p_B = self.trainSet(xBound_tf, yBound_tf, tBound_tf, W, b, lb, ub)

        u_I, v_I, p_I = self.trainSet(xInit_tf, yInit_tf, tInit_tf, W, b, lb, ub)
        
        fx, fy, fm = self.net_f(xPhisic_tf, yPhisic_tf, tPhisic_tf, W, b, Re, lb, ub)
            
        lossU = tf.reduce_mean(tf.square(u - uTrain)) \
        + tf.reduce_mean(tf.square(v - vTrain)) \
        + tf.reduce_mean(tf.square(p - pTrain))
        
        lossF = tf.reduce_mean(tf.square( lambda_f*fx )) + tf.reduce_mean(tf.square( lambda_f*fy )) + tf.reduce_mean(tf.square( fm ) )

        lossB = tf.reduce_mean(tf.square( lambda_b*(u_B - uBound) )) \
        + tf.reduce_mean(tf.square( lambda_b*(v_B - vBound) )) \
        + tf.reduce_mean(tf.square( lambda_b*(p_B - pBound )))

        lossI = tf.reduce_mean(tf.square( lambda_i*(u_I - uInit) )) \
        + tf.reduce_mean(tf.square( lambda_i*(v_I - vInit) )) \
        + tf.reduce_mean(tf.square( lambda_i*(p_I - pInit) ))

        loss = lossU + lossF + lossB + lossI     
        
        return loss
        
if __name__ == "__main__": 
# Defining variables
    D = 1 
    nu = 0.01
    Uinf = 1
    Re = nu/(Uinf*D)   
    noise = 0.0  

    # Time variables
    tStep = 0.1
    T=201
    tInit = 151 # Initial time

    # Data evaluation points      
    Ntest = 500
    Ndata = 40
    Nfis  = 10000
    Ncyl  = 10000

    # Iteration steps per method
    nIterAdam  = 10000
    nIterLBFGS = 50000
    
    # Defining Neural Network
    layers = [3]+6*[64]+[3]

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

    _, outletBConditionIdx = spatial.KDTree(Xdata).query(Xdata[(Xdata[:,0] == 15)])
    outletBConditionIdx = np.random.choice(outletBConditionIdx, 20, replace=False)
    XouBC, ouBC = conform_data(Xdata, Udata, outletBConditionIdx, T=None)

    # Boundary Conditions structure
    r = 0.5
    angle = np.pi*np.random.uniform(0, 2, Nfis)
    pts = np.array([r*np.cos(angle), r*np.sin(angle)]).T
    _, idx = spatial.KDTree(Xdata).query(pts) 
    idxCy =  np.unique(idx)
    XcyBC, CyBC = conform_data(Xdata, Udata, idxCy, T=None)

    idxBC = np.concatenate((underBConditionIdx, inletBConditionIdx, upperBConditionIdx, idxCy), axis=0, out=None)
    X_bc = np.concatenate((XunBC, XinBC, XouBC, XupBC, XcyBC), axis=0, out=None)
    U_bc = np.concatenate((UnBC, InBC, ouBC, UpBC, CyBC), axis=0, out=None) 

    # Initial Conditions
    # idxIni = select_idx(Xdata, Ninit, criterion='uni')
    X_ini = Xdata
    X_ini = np.c_[X_ini, np.zeros(X_ini.shape[0])]
    U_ini = Udata[:X_ini.shape[0],:,0]
    Ninit = X_ini.shape[0]

    # Select a number of point to test the NN
    idxTest = select_idx(Xdata, Ntest, criterion='fem')
    Xtest, Utest = conform_data(Xdata, Udata, idxTest)
    u_min_threshold = 0.001
    valid_indices = (Utest > u_min_threshold).all(axis=1)
    Xtest = Xtest[valid_indices]
    Utest = Utest[valid_indices]
    # Xtest = Xtest[(Utest[:,0]>0.01)*(Utest[:,1]>0.01)*(Utest[:,2]>0.01)] # Filter to avoide data (u, v, p) near zero
    # Utest = Utest[(Utest[:,0]>0.01)*(Utest[:,1]>0.01)*(Utest[:,2]>0.01)] # Filter to avoide data (u, v, p) near zero

    # Select training points of point to test the NN
    idxTrain = select_idx(Xdata, Ndata, criterion='fem')
    XdataTrain, UdataTrain = conform_data(Xdata, Udata, idxTrain)
    
    ptsF = np.random.uniform([-5, -5], [15, 5], size=(Nfis, 2))  #interior fis points w no data
    X_f = np.c_[ptsF, tStep*np.random.randint(T-tInit, size=Nfis) ]
    #X_f = np.vstack([X_f, XdataTrain]) #eval fis in data points

    lr = 1e-3
    
    folder = r'src/results/TestUVP'

    #Set of evaluation points
    x = np.arange(-5, 15.1, 0.1)
    y = np.arange(-5, 5.05, 0.05)
    t = np.arange(0, (T-tInit)*tStep, tStep)
    
    model = NavierStoke(XdataTrain, UdataTrain, X_f, X_bc, U_bc, X_ini, U_ini, Xtest, Utest, x, y, t, layers, lr, Re, folder, nIterAdam, nIterLBFGS)