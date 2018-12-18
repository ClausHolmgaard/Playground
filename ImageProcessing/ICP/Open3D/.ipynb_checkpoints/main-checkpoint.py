import open3d
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud as pc

from Kinect import Kinect
from Planes import find_plane, distance_from_plane
from watershed import watershed


k = Kinect(debug=True)
k.start()
k.wait_for_init()
point_cloud = k.get_pointcloud()
k.stop()

#points = pd.DataFrame(point_cloud, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
#cloud = pc(points)
#cloud.plot(IFrame_shape=(1200, 700), point_size=0.001)

pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(point_cloud[:3])

