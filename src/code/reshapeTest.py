import vtk
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from scipy import spatial
from pyDOE import lhs

def select_points(Xdata,N_u,criterion='fem'):
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

Xdata = np.load(r"src/data/VORT_DATA_VTU/Xdata.npy")
Udata = np.load(r"src/data/VORT_DATA_VTU/Udata.npy")

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
print("i")
plt.scatter(Xsort[:,0],Xsort[:,1])
plt.xlim([-5, 15])
plt.ylim([-5, 5])

print("i")