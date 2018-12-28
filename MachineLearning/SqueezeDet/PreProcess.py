import numpy as np
import cv2
import os


def closest_anchor_map(x, y, anchor_width, anchor_height, anchor_coords):
    """ Create a anchor_height x anchor_width x 3 map.
        First entry is 1 if the anchor point is closest to true point. Zero otherwise.
        Second is x offset.
        Third is y offset. """
    
    closest = 10000
    closest_x = None
    closest_y = None
    closest_x_offset = None
    closest_y_offset = None
    
    res = np.zeros((anchor_width, anchor_height, 3))
    for ix in range(anchor_width):
        for iy in range(anchor_height):
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

def get_anchors(image_width, image_height, anchor_width, anchor_height):
    """
    Generate a anchor_height x anchor_width x 2 matrix.
    Each entry is an (x, y) corrdinate mapping to image coordinates. """
    anchors = np.zeros((anchor_width, anchor_height, 2), dtype=np.uint32)
    num_anchor_nodes = anchor_height * anchor_width
    
    x_start = image_width / (anchor_width + 1)
    x_end = image_width - x_start
    y_start = image_height / (anchor_height + 1)
    y_end = image_height - y_start
    xs = np.linspace(x_start, x_end, num=anchor_width, dtype=np.uint32)
    ys = np.linspace(y_start, y_end, num=anchor_height, dtype=np.uint32)
    
    for ix in range(anchor_height):
        for iy in range(anchor_width):
            anchors[ix, iy] = (xs[ix], ys[iy])
    
    return anchors

def load_data_with_anchors(samples, data_dir, image_width, image_height, anchor_width, anchor_height):
    """
    load images
    labels will be:
    anchor_height x anchor_width x 3
        the last 3 entries is: 1 if closest gridpoint to a point. x and y offsets to closest point. """
    anchs = get_anchors(image_width, image_height, anchor_width, anchor_height)

    #with open(self.annotations_file, 'r') as f:
    #    lines = f.readlines()
    
    gt = np.zeros((len(samples), anchor_width, anchor_height, 3))
    images = np.zeros((len(samples), image_width, image_height, 3))
    
    for c, s in enumerate(samples):
        annotation_file = os.path.join(data_dir, f"{s}.an")
        image_file = os.path.join(data_dir, f"{s}.jpg")

        with open(annotation_file, 'r') as f:
            line_label = f.readline()
            obj = line_label.split(',')
            x = int(obj[0])
            y = int(obj[1])

            gt[c, :, :] = closest_anchor_map(x, y, anchor_width, anchor_height, anchs)
            im = cv2.imread(image_file)
            images[c] = im / 255.0
    
    return gt, images

def data_generator(directory, batch_size, image_width, image_height, anchor_width, anchor_height):
    
    samples = []

    # Get list of files
    for f in os.listdir(directory):
        index = int(f.split('.')[0])
        end = f.split('.')[1]
        if end == 'jpg':
            samples.append(index)
    
    samples = np.array(samples)

    while True:
        # Select files (paths/indices) for the batch
        #batch_paths = np.random.choice(a=files, size=batch_size)
        batch_samples = np.random.choice(samples, size=batch_size)

        batch_labels, batch_images = load_data_with_anchors(batch_samples,
                                                            directory,
                                                            image_width,
                                                            image_height,
                                                            anchor_width,
                                                            anchor_height)

        yield batch_images, batch_labels

def get_num_samples(data_dir):
    num_samples = 0
    for f in os.listdir(data_dir):
        end = f.split('.')[1]
        if end == 'jpg':
            num_samples += 1
    
    return num_samples