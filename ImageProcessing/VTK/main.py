import numpy as np
import Kinect

import test
import points
import AnimatedPointcloud
import PointSourceTest

#test.create_cube()
#points.create_points()
#AnimatedPointcloud.test()


k = Kinect.Kinect(debug=False)
k.start()

pc = AnimatedPointcloud.DisplayPointcloud(k.get_pointcloud)
pc.start()

print("Started")

#points = np.zeros((10000, 3))
#for k in range(len(points)):
#    points[k] = 20*(np.random.rand(3)-0.5)
#points = k.get_pointcloud()
#print("max:", np.max(points[:,1]))
#print("min:", np.min(points[:,1]))

#print("Points generated")
#pc.change_data(points)
#print("Points Added")

pc.join()
k.stop()

#PointSourceTest.test()
