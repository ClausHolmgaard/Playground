import os
import cv2
import time
import queue
import pickle
import threading
import numpy as np
from tqdm import tqdm
from shutil import copyfile


FINGER_MAP = {"Wrist": 0,
              "Thumb1": 1,
              "Thumb2": 2,
              "Thumb3": 3,
              "Thumb4": 4,
              "Index1": 5,
              "Index2": 6,
              "Index3": 7,
              "Index4": 8,
              "Middle1": 9,
              "Middle2": 10,
              "Middle3": 11,
              "Middle4": 12,
              "Ring1": 13,
              "Ring2": 14,
              "Ring3": 15,
              "Ring4": 16,
              "Pinky1": 17,
              "Pinky2": 18,
              "Pinky3": 19,
              "Pinky4": 20}

def train_validation_split(data_path, train_path, validation_path, train_samples, validation_samples, sample_type='jpg'):
    """
    Process files in data_path of the format xxxxx.sample_type.
    It will put the samples specified in train_samples into train path, and validation samples into validation path.
    It is done this way to preserve train and validation sample structure between local and remote machine.

    ALL FILES IN train_path AND validation_path WILL BE DELETED
    """
    remove_files_in_folder(train_path)
    remove_files_in_folder(validation_path)
    
    print(f"Doing train/validation split. {len(train_samples)} training samples, {len(validation_samples)} validation samples.")
    for fi in tqdm(os.listdir(data_path)):
        if fi.endswith(sample_type):
            obj = fi.split('.')
            try:
                ind = int(obj[0])
            except:
                continue
            if ind in train_samples:
                copyfile(os.path.join(data_path, fi), os.path.join(train_path, fi))
                
            if ind in validation_samples:
                copyfile(os.path.join(data_path, fi), os.path.join(validation_path, fi))
    print("")

def closest_anchor_map(x, y,
                       image_width, image_height,
                       anchor_width, anchor_height,
                       anchor_coords,
                       offset_scale):
    """ Create a anchor_height x anchor_width x 3 map.
        First entry is 1 if the anchor point is closest to true point. Zero otherwise.
        Second is x offset.
        Third is y offset. """
    #anchor_coords = get_anchors(image_width, image_height, anchor_width, anchor_height)

    x_limit = image_width / anchor_width
    y_limit = image_height / anchor_height
    dist_limit = np.sqrt(x_limit**2 + y_limit**2)

    res = np.zeros((anchor_width, anchor_height, 3))

    if x is not None and y is not None and x > 0 and y > 0:
        xs = anchor_coords[:, :, 0]
        ys = anchor_coords[:, :, 1]
        dist_matrix = np.sqrt( (xs - x)**2 + (ys - y)**2 )
        min_val = np.min(dist_matrix)
        closest_xs, closest_ys = np.where(dist_matrix<=dist_limit)
        
        # Set offsets
        for cx, cy in zip(closest_xs, closest_ys):
            anchor_x, anchor_y = anchor_coords[cx, cy]
            closest_offset_x = (x - anchor_x) / offset_scale
            closest_offset_y = (y - anchor_y) / offset_scale
            res[cx, cy, 1:] = (closest_offset_x, closest_offset_y)
        
        # Set label
        closest_x, closest_y = np.where(dist_matrix==min_val)
        closest_x = closest_x[0]  # If multiple values, the first one is used
        closest_y = closest_y[0]
        res[closest_x, closest_y, 0] = 1
        
    
    return res

def get_anchors(image_width, image_height, anchor_width, anchor_height):
    """
    Generate a anchor_height x anchor_width x 2 matrix.
    Each entry is an (x, y) corrdinate mapping to image coordinates. """
    anchors = np.zeros((anchor_width, anchor_height, 2))#, dtype=np.uint32)
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

def load_data_with_anchors(samples,
                           data_dir,
                           anno_dir,
                           image_width,
                           image_height,
                           anchor_width,
                           anchor_height,
                           offset_scale,
                           sample_type,
                           num_classes=1,
                           only_images=False,
                           greyscale=False,
                           progressbar=False):
    """
    load images
    labels will be:
    anchor_height x anchor_width x (3 * num_classes), 1 confidence score and x,y for offset.
    The first num_classes is confidence scores, then follows the offsets.
    """
    if greyscale:
        channels = 1
    else:
        channels = 3

    anchs = get_anchors(image_width, image_height, anchor_width, anchor_height)
    gt = np.zeros((len(samples), anchor_width, anchor_height, 3*num_classes))
    images = np.zeros((len(samples), image_width, image_height, channels))
    
    if progressbar:
        print(f"Loading {len(samples)} samples")
        ite = enumerate(tqdm(samples))
    else:
        ite = enumerate(samples)

    for c, s in ite:
        
        annotation_file = os.path.join(anno_dir, "%05d.an" % s)
        image_file = os.path.join(data_dir, "%05d.%s" % (s, sample_type))

        if not only_images:
            with open(annotation_file, 'r') as f:
                line_labels = f.readlines()

                point = 0
                for line_label in line_labels:
                    obj = line_label.split(',')
                    if obj[0] != '':
                        x = float(obj[0])
                        y = float(obj[1])
                    else:
                        x = None
                        y = None
                    cam = closest_anchor_map(x, y, image_width, image_height, anchor_width, anchor_height, anchs, offset_scale)
                    gt[c, :, :, point] = cam[:, :, 0]
                    gt[c, :, :, num_classes+point*2] = cam[:, :, 1]
                    gt[c, :, :, num_classes+1+point*2] = cam[:, :, 2]

                    if x is not None and y is not None:
                        point += 1
                    
        im = cv2.imread(image_file)
        if greyscale:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).reshape(image_width, image_height, 1)
        images[c] = im / 255.0

    return gt, images

def create_data_generator(directory,
                          annotations_dir,
                          batch_size,
                          image_width,
                          image_height,
                          channels,
                          anchor_width,
                          anchor_height,
                          offset_scale,
                          num_classes=1,
                          sample_type='jpg',
                          greyscale=False,
                          verbose=False,
                          queue_size=1,
                          preload_all_data=False):

    print(f"Starting data generator in: {directory}, with annotations in {annotations_dir}")
    if verbose:
        print(f"Samples: {samples}")

    samples = get_all_samples(directory, sample_type=sample_type)
    
    if preload_all_data:
        """
        all_images = []
        all_labels = []
        for s in tqdm(samples):
            data = load_data_with_anchors([s],
                                          directory,
                                          annotations_dir,
                                          image_width,
                                          image_height,
                                          anchor_width,
                                          anchor_height,
                                          offset_scale,
                                          sample_type,
                                          num_classes=num_classes,
                                          greyscale=greyscale)
            all_labels.append(data[0].reshape(anchor_width, anchor_height, 3*num_classes))
            all_images.append(data[1].reshape(image_width, image_height, 1))

        all_labels = np.array(all_labels)
        all_images = np.array(all_images)
        """
        all_labels, all_images = load_data_with_anchors(samples,
                                                        directory,
                                                        annotations_dir,
                                                        image_width,
                                                        image_height,
                                                        anchor_width,
                                                        anchor_height,
                                                        offset_scale,
                                                        sample_type,
                                                        num_classes=num_classes,
                                                        greyscale=greyscale,
                                                        progressbar=True)
        
        while True:
            ind = np.random.randint(0, len(samples), size=batch_size)

            batch_labels = all_labels[ind]
            batch_images = all_images[ind]

            yield batch_images, batch_labels
    else:
        gen = BackgroundGenerator(directory,
                                  batch_size,
                                  annotations_dir,
                                  image_width,
                                  image_height,
                                  anchor_width,
                                  anchor_height,
                                  offset_scale,
                                  sample_type,
                                  num_classes,
                                  greyscale,
                                  samples,
                                  queue_size)

        return data_generator(gen)

def data_generator(gen):
    while True:
        batch_labels, batch_images = gen.next()

        yield batch_images, batch_labels

def get_num_samples(data_dir, type_sample='jpg'):
    num_samples = 0
    for f in os.listdir(data_dir):
        end = f.split('.')[1]
        if end == type_sample:
            num_samples += 1
    
    return num_samples

def get_all_samples(data_dir, sample_type='jpg'):
    samples = []
    for fi in os.listdir(data_dir):
        if fi.endswith(sample_type):
            obj = fi.split('.')
            try:
                ind = int(obj[0])
            except:
                continue
            samples.append(ind)

    return samples

def remove_files_in_folder(folder, filetype=None):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                if filetype is not None:
                    if not file_path.endswith(filetype):
                        continue
                os.unlink(file_path)
        except Exception as e:
            print(e)

def get_hand_points(index, annotations, offset):
    """
    Dictionary with entry for each point. Each entry is (x, y, visible)
    Where visible is 1 for seen points, 0 for hidden.
    """
    this_index = annotations[index]['uv_vis']
    
    points = [None] * 21

    points[FINGER_MAP["Wrist"]] = this_index[offset + 0]

    points[FINGER_MAP["Thumb1"]] = this_index[offset + 1]
    points[FINGER_MAP["Thumb2"]] = this_index[offset + 2]
    points[FINGER_MAP["Thumb3"]] = this_index[offset + 3]
    points[FINGER_MAP["Thumb4"]] = this_index[offset + 4]

    points[FINGER_MAP["Index1"]] = this_index[offset + 5]
    points[FINGER_MAP["Index2"]] = this_index[offset + 6]
    points[FINGER_MAP["Index3"]] = this_index[offset + 7]
    points[FINGER_MAP["Index4"]] = this_index[offset + 8]

    points[FINGER_MAP["Middle1"]] = this_index[offset + 9]
    points[FINGER_MAP["Middle2"]] = this_index[offset + 10]
    points[FINGER_MAP["Middle3"]] = this_index[offset + 11]
    points[FINGER_MAP["Middle4"]] = this_index[offset + 12]

    points[FINGER_MAP["Ring1"]] = this_index[offset + 13]
    points[FINGER_MAP["Ring2"]] = this_index[offset + 14]
    points[FINGER_MAP["Ring3"]] = this_index[offset + 15]
    points[FINGER_MAP["Ring4"]] = this_index[offset + 16]

    points[FINGER_MAP["Pinky1"]] = this_index[offset + 17]
    points[FINGER_MAP["Pinky2"]] = this_index[offset + 18]
    points[FINGER_MAP["Pinky3"]] = this_index[offset + 19]
    points[FINGER_MAP["Pinky4"]] = this_index[offset + 20]

    return points

def get_left_hand(index, annotations):
    return get_hand_points(index, annotations, 0)

def get_right_hand(index, annotations):
    return get_hand_points(index, annotations, 21)

def create_rhd_annotations(annotations_file,
                           annotations_out_path,
                           color_path,
                           fingers='ALL',
                           hands_to_annotate='BOTH',
                           annotate_non_visible=True,
                           force_new_files=False):
    """
    Create annotations for RHD dataset.
    annotations_file is the file that came with the dataset.
    annotations_out_path is where the resulting annotations from this will end up.
    color_path is the path to the color images from the RHD dataset.
    fingers is an array with the fingers to annotate, or ALL for all fingers.
    hands is right, left or BOTH.
    """
    with open(annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    if force_new_files:
        remove_files_in_folder(annotations_out_path)

    print(f"Creating annotations in directory: {color_path}")
    print(f"Using annotation file: {annotations_file}")
    print(f"And outputting to: {annotations_out_path}")
    for fi in tqdm(os.listdir(color_path)):
        if fi.endswith('png'):
            anno_file_name = f"{fi.split('.')[0]}.an"
            anno_file_path = os.path.join(annotations_out_path, anno_file_name)
            ind = int(fi.split('.')[0])
            
            right_hand = get_right_hand(ind, annotations)
            left_hand = get_left_hand(ind, annotations)
            
            with open(anno_file_path, 'w') as write_file:
                if hands_to_annotate.lower() == 'right':
                    hands = [right_hand]
                elif hands_to_annotate.lower() == 'left':
                    hands = [left_hand]
                else:
                    hands = [right_hand, left_hand]

                for h in hands:
                    if fingers == 'ALL':
                        for p in h:
                            visible = p[2] != 0
                            if visible or annotate_non_visible:
                                write_file.write(f"{float(p[0])},{float(p[1])}\n")
                    else:
                        for f in fingers:
                            p = h[FINGER_MAP[f]]
                            visible = p[2] != 0
                            if visible or annotate_non_visible:
                                write_file.write(f"{float(p[0])},{float(p[1])}\n")
    print("")

class BackgroundGenerator(threading.Thread):
    def __init__(self,
                 directory,
                 batch_size,
                 annotations_dir,
                 image_width,
                 image_height,
                 anchor_width,
                 anchor_height,
                 offset_scale,
                 sample_type,
                 num_classes,
                 greyscale,
                 available_sampels,
                 queue_size=1):
        threading.Thread.__init__(self)

        self.directory = directory
        self.batch_size = batch_size
        self.annotations_dir = annotations_dir
        self.image_width = image_width
        self.image_height = image_height
        self.anchor_width = anchor_width
        self.anchor_height = anchor_height
        self.offset_scale = offset_scale
        self.sample_type = sample_type
        self.num_classes = num_classes
        self.greyscale = greyscale
        self.available_sampels = available_sampels

        self.queue = queue.Queue(maxsize=queue_size)
        self.daemon = True
        self.is_running = True
        self.start()

    def run(self):

        while self.is_running:
            batch_samples = np.random.choice(self.available_sampels, size=self.batch_size)

            self.queue.put(load_data_with_anchors(batch_samples,
                                                  self.directory,
                                                  self.annotations_dir,
                                                  self.image_width,
                                                  self.image_height,
                                                  self.anchor_width,
                                                  self.anchor_height,
                                                  self.offset_scale,
                                                  self.sample_type,
                                                  num_classes=self.num_classes,
                                                  greyscale=self.greyscale))

    def stop(self):
        self.is_running = False

    def next(self):
        return self.queue.get()

if __name__ == "__main__":
    DATA_DIR = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/color")
    TRAIN_DIR = os.path.expanduser("~/datasets/RHD/processed/train")
    VALIDATION_DIR = os.path.expanduser("~/datasets/RHD/processed/validation")
    ANNOTATIONS_PATH = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/annotations")
    RHD_ANNOTATIONS_FILE = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/anno_training.pickle")

    BATCHSIZE = 64

    HEIGHT = 320
    WIDTH = 320
    CHANNELS = 3

    anchor_width = 20
    anchor_height = 20

    #create_rhd_annotations(RHD_ANNOTATIONS_FILE, ANNOTATIONS_PATH, DATA_DIR)

    #dg = data_generator(DATA_DIR, ANNOTATIONS_PATH, BATCHSIZE, WIDTH, HEIGHT, anchor_width, anchor_height, sample_type='png')

    #print(next(dg))
    
    #import matplotlib.pyplot as plt

    #ims, bs = next(dg)

    #plt.imshow(ims[0])
    #plt.show()

    #train_validation_split(DATA_DIR, TRAIN_DIR, VALIDATION_DIR, [20, 1, 2], [10, 11, 12], sample_type='png')
    #print(get_all_samples(DATA_DIR, sample_type='png'))

    
    gen = BackgroundGenerator()
    time.sleep(2)
    gen.stop()
    time.sleep(1)
