import numpy as np

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import os
import datetime
from utilities import *

if __name__ == "__main__": 

    visualize = True

    T = 201

    Xtot = np.zeros((55456, 2))
    Utot = np.zeros((55456, 4, T))

    #Choose the vtu file
    for i in range(T):
        
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
        w = vtk_to_numpy(output.GetPointData().GetArray(3))
        
        if i == 0:
            Xtot[:,0] = x
            Xtot[:,1] = y
        
        Utot[:,0,i] = u
        Utot[:,1,i] = v
        Utot[:,2,i] = p
        Utot[:,3,i] = w

    # No need to save with duplicate points
    #np.save(r"src/data/VORT_DATA_VTU/Xtot.npy", Xtot)
    #np.save(r"src/data/VORT_DATA_VTU/Utot.npy", Utot)

    # Remove the duplicated points and save the data needed for processing
    x, idxs = np.unique(Xtot, axis=0, return_index = True)
    u = Utot[idxs,:,:]

    np.save(r"src/data/vorticityTest/Xdata.npy", x)
    np.save(r"src/data/vorticityTest/Udata.npy", u)
    
    # # Create .png files for video animation
    # if(visualize):
    #      for j in range(T):
    #          fig, ax = plt.subplots() 
    #          heatmap, _, _, _, = nonuniform_imshow(x[:,0], x[:,1], u[:,0,j])
    #          animation(heatmap, r"src/data/vorticityTest/velocidadU/u", j, 'U Field')
    #          plt.close()
    #          fig, ax = plt.subplots() 
    #          heatmap, _, _, _, = nonuniform_imshow(x[:,0], x[:,1], u[:,1,j])
    #          animation(heatmap, r"src/data/vorticityTest/velocidadV/v", j, 'V Field')
    #          plt.close()
    #          fig, ax = plt.subplots() 
    #          heatmap, _, _, _, = nonuniform_imshow(x[:,0], x[:,1], u[:,2,j], 'Pressure Field')
    #          animation(heatmap, r"src/data/vorticityTest/Presion/p", j, 'Pressure Field')
    #          plt.close()
    #          fig, ax = plt.subplots() 
    #          heatmap, _, _, _, = nonuniform_imshow(x[:,0], x[:,1], u[:,3,j], 'Vorticity Field')
    #          animation(heatmap, r"src/data/vorticityTest/Vorticity/w", j, 'Vorticity Field')
    #          plt.close()
        
    #      date = str(datetime.datetime.now().month)+str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
    #      videoCreater(r"src/data/vorticityTest/velocidadU/u", r"src/data/vorticityTest/velocidadU/u"+ str(date) + ".avi", T)
    #      videoCreater(r"src/data/vorticityTest/velocidadV/v", r"src/data/vorticityTest/velocidadV/v" + str(date) + ".avi", T)
    #      videoCreater(r"src/data/vorticityTest/Presion/p", r"src/data/vorticityTest/Presion/p" + str(date) + ".avi", T)
    #      videoCreater(r"src/data/vorticityTest/Vorticity/w", r"src/data/vorticityTest/Vorticity/w" + str(date) + ".avi", T)

    #      for k in range(T):
    #          os.remove(r"src/data/velocidadU/u" + str(k) + ".png")
    #          os.remove(r"src/data/velocidadV/v" + str(k) + ".png")
    #          os.remove(r"src/data/Presion/p" + str(k) + ".png")

    


