import os
import sys
import datetime
import numpy as np
from keras import optimizers
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

from PreProcess import *
from Models.PoolingAndFire import *
from GenerateData import generate_data


LOG_DIR = os.path.expanduser("~/logs/SqueezeDet/")
#DATA_DIR = os.path.expanduser("~/datasets/Generated")
DATA_DIR = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/color")
TRAIN_DIR = os.path.expanduser("~/datasets/RHD/processed/train")
VALIDATION_DIR = os.path.expanduser("~/datasets/RHD/processed/validation")
TRAIN_ANNOTATIONS = os.path.expanduser("~/datasets/RHD/processed/train/annotations")
VALIDATION_ANNOTATIONS = os.path.expanduser("~/datasets/RHD/processed/validation/annotations")
RHD_ANNOTATIONS_FILE = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/anno_training.pickle")
MODEL_CHECKPOINT_FILE = os.path.expanduser("~/results/SqueezeDet/model_checkpoint.h5py")
MODEL_SAVE_FILE = os.path.expanduser("~/results/SqueezeDet/model_save_done.h5py")

timestamp = '{:%Y-%m-%d_%H_%M}'.format(datetime.datetime.now())
log_folder = os.path.join(LOG_DIR, timestamp)

EPSILON = 1e-16

WEIGHT_DECAY = 0 # 0.001
#KEEP_PROB = 0.5
#CLASSES = 1

HEIGHT = 320
WIDTH = 320
CHANNELS = 1

LABEL_WEIGHT = 1.0
OFFSET_LOSS_WEIGHT = 1.0
OFFSET_SCALE = int(320 / 20) / 2

INITIAL_LR = 1e-3
OPT_DECAY = 0
DECAY_EPOCHES = 50.0
DECAY_DROP = 0.2

NUM_CLASSES = 42

VALIDATION_SPLIT = 0.1

NUM_GPU = 1
BATCHSIZE = 64

NUM_EPOCHS = 250

LIMIT_SAMPLES = None

if CHANNELS == 1:
    grey = True
else:
    grey = False

if LIMIT_SAMPLES is None:
    num_samples = get_num_samples(DATA_DIR, type_sample='png')
else:
    num_samples = LIMIT_SAMPLES

num_train_samples = int((1-VALIDATION_SPLIT) * num_samples)
num_validation_samples = int(VALIDATION_SPLIT * num_samples)

all_samples = sorted(get_all_samples(DATA_DIR, sample_type='png'))
train_samples = all_samples[:num_train_samples]
if num_validation_samples > 0:
    validation_samples = all_samples[-num_validation_samples:]
else:
    validation_samples = []

train_validation_split(DATA_DIR, TRAIN_DIR, VALIDATION_DIR, train_samples, validation_samples, sample_type='png')

create_rhd_annotations(RHD_ANNOTATIONS_FILE,
                       TRAIN_ANNOTATIONS,
                       TRAIN_DIR,
                       fingers='ALL',
                       hands_to_annotate='BOTH',
                       annotate_non_visible=True,
                       force_new_files=True)
                
create_rhd_annotations(RHD_ANNOTATIONS_FILE,
                       VALIDATION_ANNOTATIONS,
                       VALIDATION_DIR,
                       fingers='ALL',
                       hands_to_annotate='BOTH',
                       annotate_non_visible=True,
                       force_new_files=True)

#model = create_model(320, 320, 3)
model = create_model_multiple_detection(WIDTH, HEIGHT, CHANNELS, NUM_CLASSES)

out_shape = model.output_shape
anchor_width = out_shape[1]
anchor_height = out_shape[2]
print(f"Needed anchor shape: {anchor_width}x{anchor_height}")

l = create_loss_function_multiple_detection(anchor_width,
                                            anchor_height,
                                            LABEL_WEIGHT,
                                            OFFSET_SCALE,
                                            OFFSET_LOSS_WEIGHT,
                                            NUM_CLASSES,
                                            EPSILON,
                                            BATCHSIZE)

valid_choices = ['0', '1']
if os.path.exists(MODEL_CHECKPOINT_FILE):
    valid_choices.append('2')
if os.path.exists(MODEL_SAVE_FILE):
    valid_choices.append('3')

out = None
print("")
if len(valid_choices) > 2:
    while not(out in valid_choices):
        print("0: Exit")
        print("1: Fresh run")
        if '2' in valid_choices:
            print("2: Load checkpoint file")
        if '3' in valid_choices:
            print("3: Load file from completed run.")
        out = input("Selection: ")

if out == '0':
    sys.exit()
elif out == '1':
    if NUM_GPU > 1:
        model = multi_gpu_model(model, gpus=NUM_GPU)
elif out == '2':
    print(f"Loading {MODEL_CHECKPOINT_FILE}...")
    model = load_model(MODEL_CHECKPOINT_FILE, custom_objects={'loss_function': l})
elif out == '3':
    print(f"Loading {MODEL_SAVE_FILE}...")
    model = load_model(MODEL_SAVE_FILE, custom_objects={'loss_function': l})

model.summary()

opt = optimizers.Adam(lr=INITIAL_LR, decay=OPT_DECAY)
model.compile(loss=l, optimizer=opt)

print(f"Number of training samples: {num_train_samples}")
steps_epoch = num_train_samples // BATCHSIZE
if steps_epoch < 1:
    steps_epoch = 1
print(f"Steps per epoch: {steps_epoch}")

class PrintInfo(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(f"Learning rate with decay: {K.eval(lr_with_decay)}")
        #print(f"lr={K.eval(lr)}, decay={K.eval(decay)}")
        print("")

def lr_decay(epoch):
	initial_lrate = INITIAL_LR
	drop = DECAY_DROP
	epochs_drop = DECAY_EPOCHES
	lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
	return lrate

lrate = LearningRateScheduler(lr_decay)

reduce_lr_plateau = ReduceLROnPlateau(monitor='loss', 
                                      factor=0.5,
                                      patience=5,
                                      verbose=1,
                                      mode='auto',
                                      min_delta=0.01,
                                      cooldown=0,
                                      min_lr=0,)

class LRTensorBoard(TensorBoard):
    def __init__(self,
                 log_dir,
                 histogram_freq=0,
                 batch_size=BATCHSIZE,
                 write_graph=True,
                 update_freq='batch'):

        super().__init__(log_dir=log_dir,
                         histogram_freq=histogram_freq,
                         batch_size=batch_size,
                         write_graph=write_graph,
                         update_freq=update_freq)
    
    def on_batch_end(self, batch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_batch_end(batch, logs)

print_info = PrintInfo()

tensorboard = TensorBoard(log_dir=log_folder,
                          histogram_freq=0,
                          batch_size=BATCHSIZE,
                          write_graph=True,
                          write_grads=False,
                          write_images=False,
                          embeddings_freq=0,
                          embeddings_layer_names=None,
                          embeddings_metadata=None,
                          embeddings_data=None,
                          update_freq='batch')

lr_tensorboard = LRTensorBoard(log_dir=log_folder,
                               histogram_freq=0,
                               batch_size=BATCHSIZE,
                               write_graph=True,
                               update_freq='batch')

checkpoint = ModelCheckpoint(MODEL_CHECKPOINT_FILE,
                             monitor='loss',
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

train_data_gen = create_data_generator(TRAIN_DIR,
                                       TRAIN_ANNOTATIONS,
                                       BATCHSIZE,
                                       WIDTH, HEIGHT, CHANNELS,
                                       anchor_width,
                                       anchor_height,
                                       OFFSET_SCALE,
                                       num_classes=NUM_CLASSES,
                                       sample_type='png',
                                       greyscale=grey,
                                       verbose=False,
                                       queue_size=100,
                                       preload_all_data=False)

model.fit_generator(train_data_gen,
                    steps_per_epoch=steps_epoch,
                    epochs=NUM_EPOCHS,
                    verbose=1,
                    callbacks=[lrate, lr_tensorboard, checkpoint],
                    #use_multiprocessing=True,
                    #workers=4
                    )

print("Saving completed model...")
model.save(MODEL_SAVE_FILE)