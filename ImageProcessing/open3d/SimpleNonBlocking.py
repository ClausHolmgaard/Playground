import open3d
import time
from Open3DHelpers import np_to_pcd

class O3DVisualize(object):
    def __init__(self):
        self.pc = open3d.PointCloud()

        self.first_run = True


        #self.vis = open3d.Visualizer()
        self.vis = open3d.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord('k'), self.callback_test)
        self.vis.create_window()
        
    
    def update_pointcloud(self, cloud):

        #self.pc.points, self.pc.colors = self._convert_vectors(cloud)
        #self.pc = np_to_pcd(cloud)
        self.pc.points = open3d.Vector3dVector(cloud[:,:3])
        self.pc.colors = open3d.Vector3dVector(cloud[:,3:]/255.0)


        if self.first_run:
            self.vis.add_geometry(self.pc)
            self.first_run = False
        
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def stop(self):
        self.vis.destroy_window()

    def callback_test(self):
        print("DFSDFSDFSDFSF")

    def _convert_vectors(self, cloud):
        coords = cloud[:,:3]
        colors = cloud[:,3:] / 255.0

        coords_v3d = self._convert_to_v3d(coords)
        colors_v3d = self._convert_to_v3d(colors)

        return coords_v3d, colors_v3d
    
    def _convert_to_v3d(self, vector):
        o3d_v3d = open3d.Vector3dVector()
        map(o3d_v3d.append, vector)
        return o3d_v3d