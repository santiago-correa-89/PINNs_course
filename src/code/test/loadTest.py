import cv2
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def nonuniform_imshow(x, y, z, aspect=1, cmap=plt.cm.rainbow):
  # Create regular grid
  xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
  xi, yi = np.meshgrid(xi, yi)

  # Interpolate missing data
  zi = griddata((x, y), z, (xi, yi), method='linear')

  fig, ax = plt.subplots()

  hm = ax.imshow(zi, extent=[x.min(), x.max(), y.max(), y.min()]) 
  
  return hm

#Save the entire data from the VTU files.   
Xtot = np.load(r"src/data/VORT_DATA_VTU/Xtot.npy")
Utot = np.load(r"src/data/VORT_DATA_VTU/Utot.npy")

#Remove the duplicated points and save the data needed for processing
x, idxs = np.unique(Xtot, axis=0, return_index = True)
u = Utot[idxs,:,:]

np.save(r"src/data/VORT_DATA_VTU/Xdata.npy", x)
np.save(r"src/data/VORT_DATA_VTU/Udata.npy", u)

for j in range(201):
  heatmap = nonuniform_imshow(x[:,0],x[:,1],u[:,2,j])
  img = plt.colorbar(heatmap)
  plt.title('Field P (Pa)')
  plt.xlabel('coord x')
  plt.ylabel('coord y')
  plt.savefig(r"src/data/fig/Presion/P_" + str(j) + ".png")
  plt.ion()
  plt.show()
  plt.pause(1)


