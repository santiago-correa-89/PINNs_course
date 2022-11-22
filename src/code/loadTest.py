import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

def nonuniform_imshow(x, y, z, aspect=1, cmap=plt.cm.rainbow):
  # Create regular grid
  xi, yi = np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200)
  Xi, Yi = np.meshgrid(xi, yi)

  # Interpolate missing data
  inter = interpolate.interp2d(x, y, z, kind='cubic')
  zi = inter(Xi, Yi)

  fig, ax = plt.subplots(figsize=(10, 8))

  hm = ax.imshow(zi, interpolation='nearest', cmap=cmap,
                 extent=[x.min(), x.max(), y.max(), y.min()]) 
  
  ax.scatter(Xi, Yi)
  ax.set_aspect(aspect)
  return hm


Xtot = np.load(r"src/data/VORT_DATA_VTU/Xtot.npy")
Utot = np.load(r"src/data/VORT_DATA_VTU/Utot.npy")

print("subaracutanga")

x, idxs = np.unique(Xtot, axis=0, return_index = True)
u = Utot[idxs,:,:]

heatmap = nonuniform_imshow(x[:,0],x[:,1],u[:,0,100])
plt.colorbar(heatmap)
plt.show()

np.save(r"src/data/VORT_DATA_VTU/Xdata.npy", x)
np.save(r"src/data/VORT_DATA_VTU/Udata.npy", u)

#fig, ax = plt.subplots()
#ax.tricontour(x[:,0],x[:,1],u[:,0,100],levels=20,linewidths=0.5,colors='k')
#cntr2 = ax.tricontourf(x[:,0],x[:,1],u[:,0,100],levels = 20, cmap="RdBu_r")
