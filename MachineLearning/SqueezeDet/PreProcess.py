import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm

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

    if(x is not None and y is not None):
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

def load_data_with_anchors(samples, data_dir, anno_dir, image_width, image_height, anchor_width, anchor_height, sample_type):
    """
    load images
    labels will be:
    anchor_height x anchor_width x 3
        the last 3 entries is: 1 if closest gridpoint to a point. x and y offsets to closest point. """
    anchs = get_anchors(image_width, image_height, anchor_width, anchor_height)
    
    gt = np.zeros((len(samples), anchor_width, anchor_height, 3))
    images = np.zeros((len(samples), image_width, image_height, 3))
    
    for c, s in enumerate(samples):
        annotation_file = os.path.join(anno_dir, "%05d.an" % s)
        image_file = os.path.join(data_dir, "%05d.%s" % (s, sample_type))
        np_sample_file = os.path.join(anno_dir, "%05d.npy" % s)

        with open(annotation_file, 'r') as f:
            line_label = f.readline()
            obj = line_label.split(',')
            if obj[0] != '':
                x = int(obj[0])
                y = int(obj[1])
            else:
                x = None
                y = None

            if os.path.exists(np_sample_file):
                gt[c, :, :] = np.load(np_sample_file)
            else:
                gt[c, :, :] = closest_anchor_map(x, y, anchor_width, anchor_height, anchs)
                np.save(np_sample_file, gt[c, :, :])

            im = cv2.imread(image_file)
            images[c] = im / 255.0
    
    return gt, images

def data_generator(directory, annotations_dir, batch_size, image_width, image_height, anchor_width, anchor_height, sample_type='jpg'):
    
    samples = []

    # Get list of files
    for f in os.listdir(directory):
        index = int(f.split('.')[0])
        end = f.split('.')[1]
        if end == sample_type:
            samples.append(index)
    
    samples = np.array(samples)

    while True:
        # Select files (paths/indices) for the batch
        #batch_paths = np.random.choice(a=files, size=batch_size)
        batch_samples = np.random.choice(samples, size=batch_size)

        batch_labels, batch_images = load_data_with_anchors(batch_samples,
                                                            directory,
                                                            annotations_dir,
                                                            image_width,
                                                            image_height,
                                                            anchor_width,
                                                            anchor_height,
                                                            sample_type)

        yield batch_images, batch_labels

def get_num_samples(data_dir, type_sample='jpg'):
    num_samples = 0
    for f in os.listdir(data_dir):
        end = f.split('.')[1]
        if end == type_sample:
            num_samples += 1
    
    return num_samples

def remove_files_in_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def get_hand_points(index, annotations, offset):
    """
    Dictionary with entry for each point. Each entry is (x, y, visible)
    Where visible is 1 for seen points, 0 for hidden.
    """
    this_index = annotations[index]['uv_vis']
    points = {"Wrist": this_index[offset + 0],
              "Thumb1": this_index[offset + 1],
              "Thumb2": this_index[offset + 2],
              "Thumb3": this_index[offset + 3],
              "Thumb4": this_index[offset + 4],
              "Index1": this_index[offset + 5],
              "Index2": this_index[offset + 6],
              "Index3": this_index[offset + 7],
              "Index4": this_index[offset + 8],
              "Middle1": this_index[offset + 9],
              "Middle2": this_index[offset + 10],
              "Middle3": this_index[offset + 11],
              "Middle4": this_index[offset + 12],
              "Ring1": this_index[offset + 13],
              "Ring2": this_index[offset + 14],
              "Ring3": this_index[offset + 15],
              "Ring4": this_index[offset + 16],
              "Pinky1": this_index[offset + 17],
              "Pinky2": this_index[offset + 18],
              "Pinky3": this_index[offset + 19],
              "Pinky4": this_index[offset + 20]}
    
    return points

def get_left_hand(index, annotations):
    return get_hand_points(index, annotations, 0)

def get_right_hand(index, annotations):
    return get_hand_points(index, annotations, 21)

def create_rhd_annotations(annotations_file, annotations_out_path, color_path):
    with open(annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    remove_files_in_folder(annotations_out_path)

    for fi in tqdm(os.listdir(color_path)):
        if fi.endswith('png'):
            anno_file_name = f"{fi.split('.')[0]}.an"
            anno_file_path = os.path.join(annotations_out_path, anno_file_name)
            ind = int(fi.split('.')[0])
            
            right_hand = get_right_hand(ind, annotations)
            
            with open(anno_file_path, 'w') as write_file:
                # Only using index finger tip for now
                p = right_hand["Index1"]
                visible = p[2] != 0
                if visible:
                    write_file.write(f"{int(p[0])},{int(p[1])}\n")

if __name__ == "__main__":
    DATA_DIR = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/color")
    ANNOTATIONS_PATH = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/annotations")
    RHD_ANNOTATIONS_FILE = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/anno_training.pickle")

    BATCHSIZE = 64

    HEIGHT = 320
    WIDTH = 320
    CHANNELS = 3

    anchor_width = 20
    anchor_height = 20

    create_rhd_annotations(RHD_ANNOTATIONS_FILE, ANNOTATIONS_PATH, DATA_DIR)


    dg = data_generator(DATA_DIR, ANNOTATIONS_PATH, BATCHSIZE, WIDTH, HEIGHT, anchor_width, anchor_height, sample_type='png')

    #print(next(dg))
    
    import matplotlib.pyplot as plt

    ims, bs = next(dg)

    #plt.imshow(ims[0])
    #plt.show()



