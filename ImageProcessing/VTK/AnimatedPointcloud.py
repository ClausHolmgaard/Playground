#From: https://stackoverflow.com/a/44198767/6588972

import vtk
import numpy as np
import time
import threading
from vtk.util import numpy_support

class VtkPointCloud:

    def __init__(self, zMin=-255.0, zMax=255.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        #self.vtkPolyData = vtk.vtkPolyData()
        self.init_points()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

        self.vtkPolyData.GetPointData().SetScalars(self.Colors)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:3])

            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)

            #self.Colors.InsertNextTuple3(0, 255, 0)
            self.Colors.InsertNextTuple3(point[3], point[4], point[5])

        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.Colors.Modified()

    def init_points(self):

        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")

        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)

        self.vtkPolyData.GetPointData().SetScalars(self.Colors)
        self.vtkPolyData.GetPointData().SetActiveScalars('Colors')
    
    #def clear_points(self):
    #    self.vtkPoints.


class AddPointCloudTimerCallback():
    def __init__(self, renderer, iterations, cb):
        self.iterations = iterations
        self.total_iterations = 0
        self.renderer = renderer
        self.start_time = time.time()
        self.end_time = None

        self._data = None
        self._data_changed = False

        self.callback = cb
    
    def set_data(self, data):

        max_x = np.max(data[:,0])
        max_y = np.max(data[:,1])
        max_z = np.max(data[:,2])

        max_val = np.max([max_x, max_y, max_z])

        data[:,0] -= int(max_x / 2)
        data[:,1] -= int(max_y / 2)
        data[:,:3] /= max_val * 2

        data[:,3:] *= 255

        self._data = data
        self._data_changed = True

    def execute(self, iren, event):

        if self.total_iterations % 30 == 0 and self.total_iterations != 0:
            run_time = time.time() - self.start_time
            fps = self.total_iterations / run_time
            print("FPS: {}".format(fps))
            if run_time > 5:
                self.total_iterations = 0
                self.start_time = time.time()

        pointCloud = VtkPointCloud()
        self.renderer.AddActor(pointCloud.vtkActor)
        pointCloud.init_points()

        d = self.callback()
        if d is not None:
            self.set_data(d)

        if self._data_changed:
            pointCloud.init_points()
            if self._data is not None:
                for p in self._data:
                    pointCloud.addPoint(p)
            self._data_changed = False

        iren.GetRenderWindow().Render()
        if self.iterations is None:
            pass
        else:
            if self.iterations == 30:
                self.renderer.ResetCamera()
            self.iterations -= 1

        self.total_iterations += 1
        
class DisplayPointcloud(threading.Thread):
    def __init__(self, cb):
        threading.Thread.__init__(self)
        self.addPointCloudTimerCallback = None
        self.callback = cb

    def run(self):
        # Renderer
        renderer = vtk.vtkRenderer()
        #renderer.SetBackground(.2, .3, .4)
        renderer.ResetCamera()

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderWindowInteractor.Initialize()

        # Initialize a timer for the animation
        self.addPointCloudTimerCallback = AddPointCloudTimerCallback(renderer, None, self.callback)
        renderWindowInteractor.AddObserver('TimerEvent', self.addPointCloudTimerCallback.execute)
        timerId = renderWindowInteractor.CreateRepeatingTimer(30)
        self.addPointCloudTimerCallback.timerId = timerId

        # Begin Interaction
        renderWindow.SetSize(1000, 800)
        renderWindow.Render()
        renderWindowInteractor.Start()
