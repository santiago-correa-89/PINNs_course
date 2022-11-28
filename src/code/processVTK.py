import vtk
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

Xtot = np.zeros((55456, 2))
Utot = np.zeros((55456, 3, 201))

#Choose the vtu file
for i in range(201):
    
    route =r"src/data/VORT_DATA_VTU/vort_cyl_" + str(i) + ".vtu"

    # Read the source file.
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(route)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()
    num_of_points = reader.GetNumberOfPoints()
    print(f"Number of Points: {num_of_points}")

    num_of_cells = reader.GetNumberOfCells()
    print(f"Number of Cells: {num_of_cells}")

    ## 
    points = output.GetPoints()
    npts = points.GetNumberOfPoints()
    ## Each element of x is list of 3 float [xp, yp, zp]
    r = vtk_to_numpy(points.GetData())
    print(f"Shape of point data:{r.shape}")
    
    x = r[:,0]
    y = r[:,1]

    # 0->rho, 1-> rhou , 2->rhov, 3->E, 4->u, 5->v, 6->p, 7->T, 8->s
    # 9->a , 10-> Mach, 11->Sensor

    u = vtk_to_numpy(output.GetPointData().GetArray(0))
    v = vtk_to_numpy(output.GetPointData().GetArray(1))
    p = vtk_to_numpy(output.GetPointData().GetArray(2))
    
    if i == 0:
        Xtot[:,0] = x
        Xtot[:,1] = y
    
    Utot[:,0,i] = u
    Utot[:,1,i] = v
    Utot[:,2,i] = p

np.save(r"src/data/VORT_DATA_VTU/Xtot.npy", Xtot)
np.save(r"src/data/VORT_DATA_VTU/Utot.npy", Utot)

