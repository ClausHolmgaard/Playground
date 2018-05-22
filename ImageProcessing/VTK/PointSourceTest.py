import vtk
import numpy as np


def test():
    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # create source
    src = vtk.vtkPointSource()
    src.SetCenter(0, 0, 0)
    src.SetNumberOfPoints(50)

    points = np.zeros((50, 3))
    for k in range(50):
        point = 20*(np.random.rand(3)-0.5)
        
    src.SetRadius(5)
    src.Update()

    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(src.GetOutputPort())

    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # assign actor to the renderer
    ren.AddActor(actor)

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()