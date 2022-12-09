import cv2
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from PIL import Image 

from scipy import spatial
from pyDOE import lhs

import pickle

TimeStep = 0.01
############ animation and visualization ###########
def nonuniform_imshow(x, y, z, aspect=1, cmap=plt.cm.rainbow):
  # Create regular grid
  xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
  xi, yi = np.meshgrid(xi, yi)
  # Interpolate missing data
  zi = griddata((x, y), z, (xi, yi), method='cubic')
  
  fig, ax = plt.subplots()
  hm = ax.imshow(zi, extent=[x.min(), x.max(), y.max(), y.min()])
  return hm, xi, yi, zi

def animation(heatmap, folder, j, title):
  img = plt.colorbar(heatmap)
  plt.title(title)
  plt.xlabel('coord x')
  plt.ylabel('coord y')
  plt.savefig(folder + str(j) + ".png")
  plt.ion()
  plt.show()
  plt.pause(1)
  plt.close()
  return 

def videoCreater(folderIn, folderOut, T): 
    img=[]

    for i in range(T):
        img.append(cv2.imread(folderIn + str(i) + ".png"))

    height,width,layers=img[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(folderOut, fourcc, 10, (width,height))

    for j in range(T):
        video.write(img[j])

    video.release()

################# Saving and loading data structures ###################
#Pickle save
def mySave(route, variable):    
    with open(route, 'wb') as file:
        pickle.dump(variable, file)
        
#Pickle load
def myLoad(route):    
    with open(route, 'rb') as file:
        variable = pickle.load(file)
    return variable

################# Data resizing and manipulation #######################
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
    N_u = idx.shape[0]
    Xsort = Xdata[idx,:]
    time_dep = TimeStep*np.array(list(range(T))).repeat(N_u)
    X_train = np.c_[np.tile(Xsort, (T, 1)), time_dep]

    Usort = Udata[idx,:]
    U_train = np.vstack([Usort[:,:,t] for t in range(T)]) 
    return X_train, U_train

def circle_points(N_f, T=201):
    r = 0.5
    angle = np.pi* np.random.uniform(0,2,N_f)
    X_train_f = np.vstack([r*np.cos(angle),r*np.sin(angle)]).T
    X_train_f = np.c_[X_train_f, TimeStep*np.random.randint(T, size=N_f) ]#0.1 time step
    return X_train_f

#OrganizeGrid
def arangeData(x, y, t, T=None):
    if T==None:
        X = x.shape[0]
        Y = y.shape[0]
        T = t.shape[0]
    xGrid = np.c_[np.vstack(np.tile(x[:].repeat(Y), T)), np.transpose(np.vstack(np.tile([y], X*T))), np.vstack([t]).repeat(Y*X)]

    return xGrid