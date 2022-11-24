#%%
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
#import os
#os.chdir('absolute-path-to-workingDir')

#Choose the vtu file
#data ="results/out_cfns_100.vtu"
data = r"src\data\VORT_DATA_VTU\vort_cyl_5.vtu"
#data = r"C:\Users\leopo\OneDrive\Documentos\PINNs\src\data\VORT_DATA_VTU\vort_cyl_1.vtu"
# Read the source file.
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(data)
reader.Update()  # Needed because of GetScalarRange
output = reader.GetOutput()
num_of_points = reader.GetNumberOfPoints()
print(f"Number of Points: {num_of_points}")

num_of_cells = reader.GetNumberOfCells()
print(f"Number of Cells: {num_of_cells}")

## 
points = output.GetPoints()
npts = points.GetNumberOfPoints()
## Each elemnts of x is list of 3 float [xp, yp, zp]
r = vtk_to_numpy(points.GetData())
print(f"Shape of point data:{r.shape}")

## Field value Name:
n_arrays = reader.GetNumberOfPointArrays()
num_of_field = 0 
field = []
for i in range(n_arrays):
    f = reader.GetPointArrayName(i)
    field.append(f)
    print(f"Id of Field: {i} and name:{f}")
    num_of_field += 1 

print(f"Total Number of Field: {num_of_field}")


# 0->rho, 1-> rhou , 2->rhov, 3->E, 4->u, 5->v, 6->p, 7->T, 8->s
# 9->a , 10-> Mach, 11->Sensor

u = vtk_to_numpy(output.GetPointData().GetArray(0))
v = vtk_to_numpy(output.GetPointData().GetArray(1))
p = vtk_to_numpy(output.GetPointData().GetArray(2))

x = r[:,0]
y = r[:,1]
print(f"Shape of field: {np.shape(u)}")

print('u: ', u.shape)
print('r: ', r.shape)
print(np.min(u), np.max(u))


#%%
fig, ax = plt.subplots()
ax.tricontour(x,y,p,levels=14,linewidths=0.5,colors='k')
cntr2 = ax.tricontourf(x,y,p,levels = 14, cmap="RdBu_r")


fig.colorbar(cntr2, ax=ax)
ax.plot(x, y, 'ko', ms=3)
#ax2.set(xlim=(-2, 2), ylim=(-2, 2))
ax.set_title('tricontour (%d points)' % npts)

plt.subplots_adjust(hspace=0.5)

#%%
fig2, ax2 = plt.subplots()
ax2 = plt.figure().add_subplot(projection='3d')
ax2.plot(x,y,zs= vtk_to_numpy(output.GetPointData().GetArray(3)))

print("bye")