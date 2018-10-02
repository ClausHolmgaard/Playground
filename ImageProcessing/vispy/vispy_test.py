import numpy as np

from Kinect import Kinect
from VispyVisualize import Visualize


def test():
    k = Kinect()
    k.start()
    k.wait_for_init()

    p = k.get_pointcloud()
    p[:,3:] = p[:,3:] / 255.0

    k.stop()

    v = Visualize(p)
    v.run()

