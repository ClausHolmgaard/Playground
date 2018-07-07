import numpy as np

def find_plane(p1, p2, p3):
    """
    points given as np arrays
    returns a, b, c, d of plane of for ax+by+cz+d=0
    """
    v1 = p3 - p1
    v2 = p2 - p1
    
    norm = np.cross(v1, v2)
    a, b, c = norm
    
    d = -np.dot(norm, p3)
    
    return a, b, c, d

def distance_from_plane(p, a, b, c, d):
    """
    plane defined by a, b, c, d. Form: ax+by+cz+d=0
    point p is numpy array
    returns distance
    """
    norm = np.array([a, b, c])
    if np.sum(norm**2) == 0:
        return False
    return np.abs((np.dot(p, norm) + d)) / np.sqrt(np.sum(np.power(norm, 2)))

