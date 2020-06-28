import numpy as np

def concat_clouds(p1, p2):
    l = p1.shape[0] + p2.shape[0]
    new_p = np.zeros((l, 6))
    new_p[:p1.shape[0]] = p1
    new_p[p1.shape[0]:] = p2
    
    return new_p
