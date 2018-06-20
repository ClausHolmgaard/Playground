import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

class watershed(object):
    def __init__(self):
        self.watershed = None

        self.canny_args = (5, 200)
        self.kernel_size = (5, 5)
        self.transform_limit = 5
    
    def set_image(self, image):
        self.rgb = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    def set_canny_args(self, args):
        self.canny_args = args
    
    def set_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size
    
    def set_transform_limit(self, transform_limit):
        self.transform_limit = transform_limit
    
    def calc(self):
        edges = cv2.Canny(self.gray, self.canny_args[0], self.canny_args[1])
        dilate_kernel = np.ones((self.kernel_size[0], self.kernel_size[1]),np.uint8)
        dilate_edges = cv2.dilate(edges, dilate_kernel)
        
        invert_binary = cv2.bitwise_not(dilate_edges)
        
        distance_transform = cv2.distanceTransform(invert_binary, cv2.DIST_L2, 3)
        distance_transform_norm = ((distance_transform - distance_transform.min()) / (distance_transform.max() - distance_transform.min()) * 255).astype(np.uint8)
        
        DISTANCE_TRANSFORM_LIMIT = self.transform_limit

        _, distance_transform_bin = cv2.threshold(distance_transform_norm,
                                                  DISTANCE_TRANSFORM_LIMIT,
                                                  255,
                                                  cv2.THRESH_BINARY)
        
        lbl, ncc = label(distance_transform_bin)
    
        self.watershed = cv2.watershed(self.rgb, lbl)
        self.watershed[self.watershed==-1] = 0
    
    def get_regions(self, image):
        self.set_image(image)
        self.calc()
        return self.watershed
    
    def get_region_plot(self):
        pass