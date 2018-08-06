import numpy as np
import time
import matplotlib.pyplot as plt

import Open3DVisualize as vis


def example_help_function():
    import open3d
    help(open3d)
    help(open3d.PointCloud)
    help(open3d.read_point_cloud)


if __name__ == "__main__":
    #example_help_function()

    """
    k = KinectImage.Kinect()
    k.start()
    vis.visualize(k.get_pointcloud())
    k.stop()
    """

    #vis.visualize_animation()
    #v = vis.Open3DVisualize()
    #v.visualize()
    #v.start()
    #v.visualize_animation()
    #v.end()

    v = vis.Open3DVisualize()
    v.run()
    
    time.sleep(3)
    print("Calling display method")
    v.display_pc()

    while v.is_running:
        time.sleep(1)

    print("Stopping thread...")
    v.end()


    #p = KinectImage.GrabPointcloud()

    #p_nocolor = p[:,:3]

    #vis.example_help_function()
    #vis.visualize(p)
    #vis.testing()