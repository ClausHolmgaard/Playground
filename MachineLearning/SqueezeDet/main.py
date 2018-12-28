from GenerateData import generate
from Model import CustomModel
from PreProcess import PreProcess
import os


LOG_DIR = os.path.expanduser("~/logs/SqueezeDet/")
DATA_DIR = os.path.expanduser("~/datasets/Generated")
ANNOTATIONS_FILE = "annot"
anno_file = os.path.join(DATA_DIR, ANNOTATIONS_FILE)

generate(data_dir=DATA_DIR,
         annotation=anno_file,
         width=320,
         height=320,
         box_min=50,
         box_max=100,
         num_images=1000)

m = CustomModel(320, 320, 3, batchsize=256, log_dir=LOG_DIR)

pre = PreProcess(320, 320, m.anchor_width, m.anchor_height, DATA_DIR, anno_file)
labels, _, images = pre.load_data_with_anchors()

m.set_data(images, labels)
m.train('Adam')
