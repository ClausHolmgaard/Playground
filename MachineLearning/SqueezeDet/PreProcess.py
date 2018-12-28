from tqdm import tqdm
import numpy as np
import cv2
import os


class PreProcess(object):
    def __init__(self, image_width, image_height, anchor_width, anhor_height, data_dir, annotations_file):
        self.image_width = image_width
        self.image_height = image_height
        self.anchor_width = anchor_width
        self.anchor_height = anhor_height

        self.data_dir = data_dir
        self.annotations_file = annotations_file

    def get_anchors(self):
        """ Generate a anchor_height x anchor_width x 2 matrix.
        Each entry is an (x, y) corrdinate mapping to image coordinates. """
        anchors = np.zeros((self.anchor_width, self.anchor_height, 2), dtype=np.uint32)
        num_anchor_nodes = self.anchor_height * self.anchor_width
        print(f"Number of anchors: {num_anchor_nodes}")
        
        print(f"Anchor dimension: ({self.anchor_height}, {self.anchor_width})")
        print(f"Anchor shape: {anchors.shape}")
        
        x_start = self.image_width / (self.anchor_width + 1)
        x_end = self.image_width - x_start
        y_start = self.image_height / (self.anchor_height + 1)
        y_end = self.image_height - y_start
        xs = np.linspace(x_start, x_end, num=self.anchor_width, dtype=np.uint32)
        ys = np.linspace(y_start, y_end, num=self.anchor_height, dtype=np.uint32)
        
        for ix in range(self.anchor_height):
            for iy in range(self.anchor_width):
                anchors[ix, iy] = (xs[ix], ys[iy])
        
        return anchors

    def load_data(self):

        print(f"Loading data in: {self.data_dir}")
        print(f"With annotations file: {self.annotations_file}")

        with open(annotation, 'r') as f:
            lines = f.readlines()
        
        gt = [(None, None)] * len(lines)
        
        for l in lines:
            obj = l.split(',')
            pic_id = int(obj[0].split('.')[0])
            x = int(obj[1])
            y = int(obj[2])
            
            gt[pic_id] = (x, y)

        images = []
        
        for fi in os.listdir(DATA_DIR):
            if not fi.endswith('jpg'):
                continue
            im = cv2.imread(os.path.join(DATA_DIR, fi))
            images.append(im)
        
        return gt, images

    def closest_anchor_map(self, x, y, anchor_coords):
        """ Create a anchor_height x anchor_width x 3 map.
            First entry is 1 if the anchor point is closest to true point. Zero otherwise.
            Second is x offset.
            Third is y offset. """
        
        closest = 10000
        closest_x = None
        closest_y = None
        closest_x_offset = None
        closest_y_offset = None
        
        res = np.zeros((self.anchor_width, self.anchor_height, 3))
        for ix in range(self.anchor_width):
            for iy in range(self.anchor_height):
                p_x, p_y = anchor_coords[ix, iy]
                dist = np.sqrt( (x - p_x)**2 + (y - p_y)**2 )
                if dist < closest:
                    closest = dist
                    closest_x = ix
                    closest_y = iy
                    closest_x_offset = (x - p_x)
                    closest_y_offset = (y - p_y)
        
        res[closest_x, closest_y, 0] = 1
        res[closest_x, closest_y, 1:] = (closest_x_offset, closest_y_offset)
        
        return res

    def load_data_with_anchors(self):
        """
        load images
        labels will be:
        anchor_height x anchor_width x 3
            the last 3 entries is: 1 if closest gridpoint to a point. x and y offsets to closest point.
        """
        anchs = self.get_anchors()

        with open(self.annotations_file, 'r') as f:
            lines = f.readlines()
        
        gt = np.zeros((len(lines), self.anchor_width, self.anchor_height, 3))
        gt_clean = [(None, None)] * len(lines)
        images = np.zeros((len(lines), self.image_width, self.image_height, 3))
        
        for c, l in enumerate(tqdm(lines)):
            obj = l.split(',')
            pic_id = int(obj[0].split('.')[0])
            x = int(obj[1])
            y = int(obj[2])
            
            gt[pic_id, :, :] = self.closest_anchor_map(x, y, anchs)
            gt_clean[pic_id] = (x, y)
        
        for fi in tqdm(os.listdir(self.data_dir)):
            if not fi.endswith('jpg'):
                continue
            im = cv2.imread(os.path.join(self.data_dir, fi))
            i = int(fi.split('.')[0])
            images[i] = im / 255.0
        
        images = np.array(images)
        
        return gt, gt_clean, images