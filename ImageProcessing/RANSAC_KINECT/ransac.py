"""
Perform ransac on a pointcloud.

return:
n best matching planes, and corresponding points.
"""

import numpy as np
from tqdm import tqdm
from Planes import find_plane, distance_from_plane

def multidim_isin_mask(arr1, arr2):
    arr1_view = arr1.view([('', arr1.dtype)] * arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)] * arr2.shape[1])
    isin = np.isin(arr1_view, arr2_view)
    return isin

def remove_points_from_cloud(cloud, points):
    t = multidim_isin_mask(cloud, points)
    tt = np.repeat(t, cloud.shape[1], axis=1)
    c = cloud[~tt].reshape((-1, cloud.shape[1]))
    return c

def get_inliers(p1, p2, p3, points, limit):
    p_a, p_b, p_c, p_d = find_plane(p1, p2, p3)
    
    dists = distance_from_plane(points, p_a, p_b, p_c, p_d)
    if dists is False:
        return False
    
    inliers = points[dists<limit]
    
    res = {"Plane": [p_a, p_b, p_c, p_d],
           "num_inliers": inliers.shape[0],
           "inliers": inliers,
           "dists": dists}

    return res

def get_rand_points(points):
    
    r = np.random.choice(range(len(points)), 3)
    p = points[r]

    return p

def do_ransac(pc, iterations, limit, min_inliers, stop_at):
    best_fit = []
    
    ps = pc[:,:3]
    ps_select = pc[:,:3]
    
    for ite in tqdm(range(iterations)):
        if(ps_select.shape[0] < stop_at):
            break
        p1, p2, p3 = get_rand_points(ps_select)
       
        fit = get_inliers(p1, p2, p3, ps, limit)
        if fit is False:
            continue
        
        if fit["num_inliers"] > min_inliers:
            best_fit.append(fit)
            ps_select = remove_points_from_cloud(np.ascontiguousarray(ps_select), np.ascontiguousarray(fit["inliers"]))
        
    return best_fit

def sort_fits(fits):
    fits = sorted(fits, key=lambda k: k['num_inliers']) 
    #for r in fits:
    #    print("Number of inliers: {}".format(r["num_inliers"]))
    return fits

def color_pointcloud(cloud, fits, lim):
    pc_color = np.copy(cloud)

    if len(fits) > 0:
        cond_blue = np.less(fits[-1]["dists"], lim)
        np.putmask(pc_color[:,5], cond_blue, 1.0)


    if len(fits) > 1:
        cond_green = np.less(fits[-2]["dists"], lim)
        np.putmask(pc_color[:,4], cond_green, 1.0)

    if len(fits) > 2:
        cond_red = np.less(fits[-3]["dists"], lim)
        np.putmask(pc_color[:,3], cond_red, 1.0)
    
    return pc_color

def ransac(pc, iterations, limit=1, min_inliers=500, stop_at=2000):
    fits = do_ransac(pc, iterations, limit, min_inliers, stop_at)
    sorted_fits = sort_fits(fits)
    colored_cloud = color_pointcloud(pc, fits, limit)

    return colored_cloud