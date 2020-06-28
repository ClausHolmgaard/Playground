import numpy as np
import pandas as pd
import time
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from pyntcloud import PyntCloud as pc
from tqdm import tqdm_notebook as tqdm
import numba as nb

from Kinect import Kinect
from Planes import find_plane, distance_from_plane
from watershed import watershed

k = Kinect(debug=True)
k.start()
k.wait_for_init()
point_cloud = k.get_pointcloud()
k.stop()

points = pd.DataFrame(point_cloud.astype(np.float32), columns=['x', 'y', 'z', 'red', 'green', 'blue'])
cloud = pc(points)
#cloud.plot(IFrame_shape=(1200, 700), point_size=0.001)
cloud.plot(width=900, initial_point_size=0.01)

time.sleep(5)