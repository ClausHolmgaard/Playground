import vispy_example
import vispy_test

from Kinect import Kinect

"""
k = Kinect(debug=False)
k.start()

vis_ex = vispy_example.VispyExample(k.get_pointcloud)
vis_ex.run()

k.stop()
"""

vispy_test.test()

