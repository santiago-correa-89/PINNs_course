import vtk
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from scipy import spatial
from pyDOE import lhs

def select_idx(Xdata,N_u,criterion='fem'):
    xmin = -5
    xmax = 15
    ymin = -5
    ymax = 5
    if(criterion=='fem'): #random sampling in fem domain. more chances where finer grid
        idx = np.random.choice(Xdata.shape[0],N_u, replace=False)
    if(criterion=='uni'): #random sampling in variable space. uniform (x,y) in ([-5,15],[-5,5])
        pts = np.random.uniform([xmin, ymin], [xmax,ymax], size=(N_u,2))
        _, idx = spatial.KDTree(Xdata).query(pts) 
    if(criterion=='lhs'):
        samps = lhs(2,N_u) #returns in range [0,1]x[0,1]. Rescale
        samps[:,0]*=(xmax-xmin)
        samps[:,0]+=xmin
        samps[:,1]*=(ymax-ymin)
        samps[:,1]+=ymin
        _, idx = spatial.KDTree(Xdata).query(samps) 
    return idx

def conform_data(Xdata, Udata, idx, T=None):
    if T==None:
        T = Udata.shape[-1]
    Xsort = Xdata[idx,:]
    X_train = np.c_[np.vstack([Xsort]*T),np.array(list(range(T))).repeat(N_u)]

    Usort = Udata[idx,:]
    U_train = np.vstack([Usort[:,:,t] for t in range(T)]) 
    return X_train, U_train

def circle_points(N_f):
    r = 0.5
    angle = np.pi* np.random.uniform(0,2,N_f)
    X_train_f = np.vstack([r*np.cos(angle),r*np.sin(angle)]).T
    return X_train_f

Xdata = np.load(r"src/data/VORT_DATA_VTU/Xdata.npy")
Udata = np.load(r"src/data/VORT_DATA_VTU/Udata.npy")

""" version without function definitions
N_u = 100 #num of random samples to take
T = Udata.shape[-1] # T = 201 
T = 5 #test low dim. comment later

idx = select_points(Xdata,N_u,criterion='uni') #take N_u number of random indexes

Xsort = Xdata[idx,:]
X_train = np.c_[np.vstack([Xsort]*T),np.array(list(range(T))).repeat(N_u)] #add 000011112222...201,201,201,201
#Xtrain = np.vstack([Xsort]*T) #vstack all (x,y) T times
#Xtrain = np.c_[Xtrain,np.array(list(range(T))).repeat(N_u)] #add 000011112222...201,201,201,201

Usort = Udata[idx,:,:T]
U_train = np.vstack([Usort[:,:,t] for t in range(T)]) #unroll vertically
#Usort = np.vstack([Usort[:,:,t] for t in range(T)]) #unroll vertically
N_f = 1500
angle = np.pi* np.random.uniform(0,2,N_f)
r = 0.5
X_train_f = np.vstack([r*np.cos(angle),r*np.sin(angle)]).T
 """
N_u = 100
N_f = 1500
idx = select_idx(Xdata, N_u, criterion='lhs')
X_d_train, U_d_train = conform_data(Xdata, Udata, idx)
X_f_train = circle_points(N_f)

print("i")
plt.scatter(Xdata[idx,0],Xdata[idx,1])
plt.scatter(X_f_train[:,0],X_f_train[:,1],c='red')
plt.legend(['Datos', 'Física'])
plt.xlim([-5, 15])
plt.ylim([-5, 5])

print("i")