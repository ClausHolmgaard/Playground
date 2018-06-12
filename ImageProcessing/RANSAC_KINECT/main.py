from Visualize import Vispy
from Kinect import Kinect
import ransac
import threading
import time


class RansacInterface(object):
    def __init__(self, kinect_instance):
        #threading.Thread.__init__(self)
        self.k = kinect_instance
        self.pc = None

        #self.cloud_lock = threading.Lock()

        #self._is_running = True
    
    """
    def run(self):
        while self._is_running:
            self.do_ransac()
        
        print("Done!")

    def stop(self):
        self._is_running = False
    """

    def do_ransac(self):
        
        point_cloud = k.get_pointcloud()
        point_cloud = ransac.ransac(point_cloud, 300, 10, 8000)
        #with self.cloud_lock:
        #    self.pc = point_cloud
        return point_cloud

    def get_pointcloud(self, ransac):
        if ransac:
            return self.do_ransac()
        else:
            return k.get_pointcloud()
        #with self.cloud_lock:
        #    return self.pc



k = Kinect(debug=False)
#k.scale_size(200)
k.scale_rgb(True)
k.start()

ri = RansacInterface(k)
#ri.start()

vis = Vispy(ri.get_pointcloud, point_size=0.004, edge_width=0.004, symbol='o')
#vis = Vispy(k.get_pointcloud)
vis.run()

#ri.stop()
#ri.join()
k.stop()