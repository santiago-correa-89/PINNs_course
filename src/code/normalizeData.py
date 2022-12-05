import os
import vtk
import scipy
from scipy.interpolate import griddata
from pyDOE import lhs

# Plot commands
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotting

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import datetime
import pickle

from reshapeTest import *
from loadData import *
from PINNs import *
from videoCreater import *

#Data Normalization
def myNormalize(X):
    min_values = X.min(axis=0)
    max_values = X.max(axis=0)
    X = ((X - min_values)/(max_values - min_values))*2 - 1
    return X

# Load Data Xdata refers to spacial position of point, Udata is the Velocity field and Pressure fields for the points. 
Xdata = np.load(r"src/data/VORT_DATA_VTU/Xdata.npy")
Udata = np.load(r"src/data/VORT_DATA_VTU/Udata.npy")

Xdata, Udata = conform_data(Xdata, Udata, np.arange(Xdata.shape[0]))

XdataNorm = myNormalize(Xdata)
UdataNorm = myNormalize(Udata)

np.save(r"src/data/VORT_DATA_VTU/XdataNorm.npy", XdataNorm)
np.save(r"src/data/VORT_DATA_VTU/UdataNorm.npy", UdataNorm)