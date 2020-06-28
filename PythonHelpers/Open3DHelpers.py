import open3d
import pandas as pd
import numpy as np
import copy


def np_to_pcd(np_pc, draw=False):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(np_pc[:,:3])
    pcd.colors = open3d.Vector3dVector(np_pc[:,3:]/255.0)

    if draw:
        open3d.draw_geometries([pcd])

    return pcd

def pcd_to_np(pcd, transform=None):
    pcd_temp = copy.deepcopy(pcd)
    if transform is not None:
        pcd_temp.transform(transform)
    p = np.asarray(pcd_temp.points)
    c = np.asarray(pcd_temp.colors)
    pc = np.zeros((p.shape[0], 6))
    pc[:,:3] = p
    pc[:,3:] = c * 255.0
    
    return pc