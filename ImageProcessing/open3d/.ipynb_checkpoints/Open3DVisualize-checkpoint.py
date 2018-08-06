import open3d
import numpy as np
import time
import KinectImage

def example_help_function():
    #help(open3d)
    help(open3d.PointCloud)
    help(open3d.read_point_cloud)

def visualize(pointcloud):
    
    d = pointcloud[:,:3]
    c = pointcloud[:,3:]
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(d)
    pcd.colors = open3d.Vector3dVector(c)

    vis = open3d.Visualizer()
    vis.create_window()
    
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def visualize_animation():

    

    k = KinectImage.Kinect()
    k.start()
    
    pcd = open3d.PointCloud()
    
    vis = open3d.Visualizer()

    

    vis.create_window()
    
    before = time.time()

    for i in range(10):
        pointcloud = k.get_pointcloud()
        d = pointcloud[:,:3]
        c = pointcloud[:,3:]
        pcd.points = open3d.Vector3dVector(d)
        pcd.colors = open3d.Vector3dVector(c)
        vis.add_geometry(pcd)
        vis.update_geometry()
        vis.reset_view_point(True)
        vis.poll_events()
        vis.update_renderer()
    after = time.time()
    print("Loop time:", after-before)
    k.stop()



def testing():
    f = r'/home/clh/git/compile/Open3D/src/Test/TestData/fragment.ply'

    print("Load a ply point cloud, print it, and render it")
    pcd = open3d.read_point_cloud(f)
    print(pcd)
    print(np.asarray(pcd.colors).shape)
    #open3d.draw_geometries([pcd])
    