#From: https://stackoverflow.com/a/44198767/6588972

import vtk
import numpy as np
import time
import threading
from vtk.util import numpy_support

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            #pointId = self.vtkPoints.In
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
            #print(self.vtkPoints.GetData())
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()
    
    def setData(self, data):
        self.vtkPoints.SetData(data)

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


class AddPointCloudTimerCallback():
    def __init__(self, renderer, iterations):
        self.iterations = iterations
        self.total_iterations = 0
        self.renderer = renderer
        self.start_time = time.time()
        self.end_time = None

    def execute(self, iren, event):
        if self.iterations is None:
            pass
        else:
            pass

        if self.total_iterations % 30 == 0 and self.total_iterations != 0:
            #iren.DestroyTimer(self.timerId)
            run_time = time.time() - self.start_time
            fps = self.total_iterations / run_time
            print("FPS: {}".format(fps))
            #self.start_time = time.time()

        pointCloud = VtkPointCloud()
        self.renderer.AddActor(pointCloud.vtkActor)
        pointCloud.clearPoints()
        
        
        points = np.zeros((10000, 3))
        for k in range(10000):
            point = 20*(np.random.rand(3)-0.5)
            pointCloud.addPoint(point)
            #points[k] = 20*(np.random.rand(3)-0.5)
        
        #points = points.reshape(3, -1)
        #points_vtk = numpy_support.numpy_to_vtk(points.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        #pointCloud.addPoint(points_vtk)
        #pointCloud.setData(points_vtk)

        #pointCloud.addPoint([0,0,0])
        #pointCloud.addPoint([0,0,0])
        #pointCloud.addPoint([0,0,0])
        #pointCloud.addPoint([0,0,0])
        iren.GetRenderWindow().Render()
        if self.iterations is None:
            pass
        else:
            if self.iterations == 30:
                self.renderer.ResetCamera()
            self.iterations -= 1
            #print("Iteration {}".format(self.iterations))

        self.total_iterations += 1
        
class DisplayPointcloud(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(.2, .3, .4)
        renderer.ResetCamera()

        # Render Window
        renderWindow = vtk.vtkRenderWindow()

        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderWindowInteractor.Initialize()

        # Initialize a timer for the animation
        addPointCloudTimerCallback = AddPointCloudTimerCallback(renderer, None)
        renderWindowInteractor.AddObserver('TimerEvent', addPointCloudTimerCallback.execute)
        timerId = renderWindowInteractor.CreateRepeatingTimer(20)
        addPointCloudTimerCallback.timerId = timerId

        # Begin Interaction
        renderWindow.SetSize(1000, 800)
        renderWindow.Render()
        renderWindowInteractor.Start()
