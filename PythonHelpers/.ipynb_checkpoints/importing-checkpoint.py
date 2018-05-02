import h5py
import numpy as np
import cv2

#No thinking required, thank you interwebs!
#http://ddokkddokk.tistory.com/21

def load_nyu_images(filepath, im_num, norm_depth=True):
    with h5py.File(filepath, 'r') as data:
        img = data['images'][im_num]
        img_ = np.empty([480, 640, 3])
        img_[:,:,0] = img[0,:,:].T
        img_[:,:,1] = img[1,:,:].T
        img_[:,:,2] = img[2,:,:].T
        img_ = img_.astype('float32')
        img_ = img_/255.0

        depth = data['depths'][im_num]
        depth_ = np.empty([480, 640, 3])
        depth_[:,:,0] = depth[:,:].T
        depth_[:,:,1] = depth[:,:].T
        depth_[:,:,2] = depth[:,:].T

        if norm_depth:
            # Normalize depth
            depth_ = (depth_ - np.min(depth_)) / (np.max(depth_) - np.min(depth_))
    
        raw_depth = data['rawDepths'][im_num]
        raw_depth_ = np.empty([480, 640, 3])
        raw_depth_[:,:,0] = raw_depth[:,:].T
        raw_depth_[:,:,1] = raw_depth[:,:].T
        raw_depth_[:,:,2] = raw_depth[:,:].T

        if norm_depth:
            # Normalize depth
            raw_depth_ = (raw_depth_ - np.min(raw_depth_)) / (np.max(raw_depth_) - np.min(raw_depth_))
    
    return img_, raw_depth_, depth_