import vispy_example
from Kinect import Kinect

k = Kinect()
k.start()

vis_ex = vispy_example.VispyExample(k.get_pointcloud)
vis_ex.run()

k.stop()