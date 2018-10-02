import numpy as np
import time
import matplotlib.pyplot as plt
from Kinect import Kinect

#import Open3DVisualize as vis
from SimpleNonBlocking import O3DVisualize

def example_help_function():
    import open3d
    help(open3d)
    help(open3d.PointCloud)
    help(open3d.read_point_cloud)


if __name__ == "__main__":
    #example_help_function()

    
    k = Kinect()
    k.start()
    k.wait_for_init()

    o3d_vis = O3DVisualize()

    for i in range(10):
        o3d_vis.update_pointcloud(k.get_pointcloud())
        #time.sleep(0.5)
    
    o3d_vis.stop()
    k.stop()

    #vis.visualize_animation()
    #v = vis.Open3DVisualize()
    #v.visualize()
    #v.start()
    #v.visualize_animation()
    #v.end()


    """
    v = vis.Open3DVisualize()
    v.run()
    
    time.sleep(3)
    print("Calling display method")
    v.display_pc()

    while v.is_running:
        time.sleep(1)

    print("Stopping thread...")
    v.end()
    """

    #p = KinectImage.GrabPointcloud()

    #p_nocolor = p[:,:3]

    #vis.example_help_function()
    #vis.visualize(p)
    #vis.testing()