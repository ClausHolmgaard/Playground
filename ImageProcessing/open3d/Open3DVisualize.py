# IMPORTANT:
# Vector3dVector is very slow right now, see: https://github.com/IntelVCL/Open3D/issues/403

import open3d
import numpy as np
import time

from Kinect import Kinect


class Open3DVisualize():
    def __init__(self):
        self.k = Kinect(debug=True)
        self.k.start()
        self.k.wait_for_init()

        self.is_running = True

        self.pcd1 = open3d.PointCloud()
        self.pcd2 = open3d.PointCloud()

        self.vis = open3d.Visualizer()
        #self.vis = open3d.VisualizerWithKeyCallback()
        #self.vis.register_key_callback(ord('k'), self.callback_test)
        self.vis.create_window()
        self.vis.add_geometry(self.pcd1)
        self.vis.add_geometry(self.pcd2)
        


        #self.display_pc()
        #print(f"GEOMS: {len(self.geoms)}")
        #self.draw()
    
    def end(self):
        self.is_running = False
        self.k.stop()
    
    def run(self):
        #while self.is_running:
        #    time.sleep(1)
        #    print("ite")
        
        #print("Stopping...")
        #self.end()
        self.draw()
    
    def draw(self):
        key_to_callback = {}
        key_to_callback[ord("K")] = self.callback_test
        #open3d.draw_geometries_with_key_callbacks(self.geoms, key_to_callback)
        #open3d.draw_geometries_with_key_callbacks([self.pcd1, self.pcd2], key_to_callback)
    
    def display_pc(self):
        print("Displayng PC")
        pointcloud = self.k.get_pointcloud()
        d = pointcloud[:,:3]
        c = pointcloud[:,3:] / 255.0
        #pcd = open3d.PointCloud()
        self.pcd1.points = open3d.Vector3dVector(d)
        self.pcd1.colors = open3d.Vector3dVector(c)

        #self.geoms.append(pcd)

        #self.vis.add_geometry(pcd)
        #self.geoms.append(pcd)
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()


    def get_pc(self):
        print("Grabbing pointcloud")
        return self.k.get_pointcloud()


    def visualize_animation(self):
            
        self.vis.update_geometry()
        self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()
    
            

    def callback_test(self, vis):
        self.is_running = False
        print("Callback!")
        print(dir(vis))

        #d = np.asarray(vis.capture_depth_float_buffer())
        #print(d.shape)

    def visualize(self):
        
        pointcloud = self.k.get_pointcloud()
        d = pointcloud[:,:3]
        c = pointcloud[:,3:] / 255.0
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(d)
        pcd.colors = open3d.Vector3dVector(c)

        """
        vis = open3d.Visualizer()
        vis.create_window()
        
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
        """

        key_to_callback = {}
        key_to_callback[ord("K")] = self.callback_test
        open3d.draw_geometries_with_key_callbacks([pcd], key_to_callback)

        #open3d.draw_geometries([pcd])




def testing():
    f = r'/home/clh/git/compile/Open3D/src/Test/TestData/fragment.ply'

    print("Load a ply point cloud, print it, and render it")
    pcd = open3d.read_point_cloud(f)
    print(pcd)
    print(np.asarray(pcd.colors).shape)
    #open3d.draw_geometries([pcd])
    