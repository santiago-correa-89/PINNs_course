import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

#Choose the vtu file
data ="results/out_cfns_100.vtu"

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
x = vtk_to_numpy(points.GetData())
print(f"Shape of point data:{x.shape}")

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

u = vtk_to_numpy(output.GetPointData().GetArray(4))
print(f"Shape of field: {np.shape(u)}")

print('u: ', u.shape)
print('x: ', x.shape)
print(np.min(u), np.max(u))