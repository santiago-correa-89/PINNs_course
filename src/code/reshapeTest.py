import vtk
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

Xdata = np.load(r"src/data/VORT_DATA_VTU/Xdata.npy")
Udata = np.load(r"src/data/VORT_DATA_VTU/Udata.npy")

N_u = 4 #num of random samples to take
T = Udata.shape[-1] # T = 201 
T = 5 #test low dim. comment later

idx = np.random.choice(Xdata.shape[0],N_u, replace=False) #take N_u number of random indexes

Xsort = Xdata[idx,:]
Xsort = np.vstack([Xsort]*T) #vstack all (x,y) T times
Xsort = np.c_[Xsort,np.array(list(range(T))).repeat(N_u)] #add 000011112222...201,201,201,201

Usort = Udata[idx,:,:T]
Usort = np.vstack([Usort[:,:,t] for t in range(T)]) #unroll vertically
print("i")
print(np.array_equal(Unew,Ualt))