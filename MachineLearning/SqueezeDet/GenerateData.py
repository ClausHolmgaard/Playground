import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm


def generate_data(data_dir=r"./data",
                  width=320,
                  height=320,
                  box_min=50,
                  box_max=100,
                  num_images=100):

    print(f"Generating data in: {data_dir}")

    remove_files_in_folder(data_dir)

    for i in range(num_images):
        im_file_name = f"{i}.jpg"
        anno_file_name = os.path.join(data_dir, f"{i}.an")
        im, center = generate_random_boximage(width, height, box_min, box_max)
        with open(anno_file_name, 'w') as f:
            f.write(f"{center[0]},{center[1]}\n")
        im_to_save = im
        cv2.imwrite(os.path.join(data_dir, im_file_name), im)
    
def generate_random_boximage(width, height, box_min, box_max):
    im = np.zeros((width, height, 3), dtype=np.float32)

    w = np.random.randint(box_min, box_max)
    h = np.random.randint(box_min, box_max)
    x = np.random.randint(w, width-w)
    y = np.random.randint(h, height-h)

    im[x:x+w,y:y+h] = (0, 0, 255)
    
    center_x = int(x + w/2)
    center_y = int(y + h/2)
    
    return im, (center_x, center_y)

def remove_files_in_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)