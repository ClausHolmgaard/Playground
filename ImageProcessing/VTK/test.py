import vtk

def create_cube():
    print("Creating cube...")
    cube = vtk.vtkCubeSource()

    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputConnection(cube.GetOutputPort())
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    cube_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.AddActor(cube_actor)
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("Simple VTK scene")
    render_window.SetSize(400, 400)
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    interactor.Initialize()
    render_window.Render()
    interactor.Start()

    print("Done!")