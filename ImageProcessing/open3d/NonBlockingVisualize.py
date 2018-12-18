# IMPORTANT:
# Vector3dVector is very slow right now, see: https://github.com/IntelVCL/Open3D/issues/403

import open3d
import numpy as np
import time

from Kinect import Kinect

class NonBlockVis(object):
    def __init__(self):
        self.k = Kinect(debug=True)
        self.k.start()
        self.k.wait_for_init()
        
        self.pcd = open3d.PointCloud()
        
        self.vis = open3d.Visualizer()
        self.vis.create_window()
        #self.vis.add_geometry(self.pcd)
        #self.vis.run()

    def show(self):
        pointcloud = self.k.get_pointcloud()
        print(f"pointcloud shape: {pointcloud.shape}")
        d = pointcloud[:,:3]
        c = pointcloud[:,3:] / 255.0
        self.pcd.points = open3d.Vector3dVector(d)
        self.pcd.colors = open3d.Vector3dVector(c)

        self.vis.add_geometry(self.pcd)

        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.run()
        
        
    
    def stop(self):
        self.k.stop()

if __name__ == "__main__":

    v = NonBlockVis()
    time.sleep(2)
    v.show()

    for _ in range(5):
        time.sleep(1)
    
    v.stop()
