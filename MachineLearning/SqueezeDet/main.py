import numpy as np
import os
import datetime
from keras import optimizers
from keras import backend as K
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.utils import multi_gpu_model

from Models.PoolingAndFire import *
from PreProcess import *
from GenerateData import generate_data


LOG_DIR = os.path.expanduser("~/logs/SqueezeDet/")
#DATA_DIR = os.path.expanduser("~/datasets/Generated")
DATA_DIR = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/color")
TRAIN_DIR = os.path.expanduser("~/datasets/RHD/processed/train")
VALIDATION_DIR = os.path.expanduser("~/datasets/RHD/processed/validation")
ANNOTATIONS_PATH = os.path.expanduser("~/datasets/RHD/processed/train/annotations")
RHD_ANNOTATIONS_FILE = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/anno_training.pickle")
MODEL_SAVE_FILE = os.path.expanduser("~/results/SqueezeDet/model_checkpoint.h5py")

timestamp = '{:%Y-%m-%d_%H_%M}'.format(datetime.datetime.now())
log_folder = os.path.join(LOG_DIR, timestamp)

EPSILON = 1e-16

WEIGHT_DECAY = 0 # 0.001
KEEP_PROB = 0.5
CLASSES = 1

LABEL_WEIGHT = 1.0
OFFSET_LOSS_WEIGHT = 1.0
OFFSET_WEIGHT = 40.0

HEIGHT = 320
WIDTH = 320
CHANNELS = 3

INITIAL_LR = 1e-4
OPT_DECAY = 1e-5
DECAY_EPOCHES = 2.0
DECAY_DROP = 0.8

NUM_CLASSES = 42

LIMIT_SAMPLES = None

VALIDATION_SPLIT = 0.1

NUM_GPU = 4
BATCHSIZE = 64

#generate_data(DATA_DIR, WIDTH, HEIGHT, box_min=50, box_max=100, num_images=1000)

if LIMIT_SAMPLES is None:
    num_samples = get_num_samples(DATA_DIR, type_sample='png')
else:
    num_samples = LIMIT_SAMPLES

num_train_samples = int((1-VALIDATION_SPLIT) * num_samples)
num_validation_samples = int(VALIDATION_SPLIT * num_samples)

all_samples = sorted(get_all_samples(DATA_DIR, sample_type='png'))
train_samples = all_samples[:num_train_samples]
validation_samples = all_samples[-num_validation_samples:]

train_validation_split(DATA_DIR, TRAIN_DIR, VALIDATION_DIR, train_samples, validation_samples, sample_type='png')

create_rhd_annotations(RHD_ANNOTATIONS_FILE,
                       ANNOTATIONS_PATH,
                       TRAIN_DIR,
                       fingers='ALL',
                       hands_to_annotate='BOTH',
                       annotate_non_visible=False,
                       force_new_files=True)

#model = create_model(320, 320, 3)

model = create_model_multiple_detection(WIDTH, HEIGHT, CHANNELS, NUM_CLASSES)

out_shape = model.output_shape
anchor_width = out_shape[1]
anchor_height = out_shape[2]
print(f"Needed anchor shape: {anchor_width}x{anchor_height}")

model.summary()

#l = create_loss_function(anchor_width,
#                         anchor_height,
#                         LABEL_WEIGHT,
#                         OFFSET_WEIGHT,
#                         OFFSET_LOSS_WEIGHT,
#                         EPSILON,
#                         BATCHSIZE)
l = create_loss_function_multiple_detection(anchor_width,
                                            anchor_height,
                                            LABEL_WEIGHT,
                                            OFFSET_WEIGHT,
                                            OFFSET_LOSS_WEIGHT,
                                            NUM_CLASSES,
                                            EPSILON,
                                            BATCHSIZE)

print(f"Number of training samples: {num_train_samples}")
steps_epoch = num_train_samples // BATCHSIZE
if steps_epoch < 1:
    steps_epoch = 1
print(f"Steps per epoch: {steps_epoch}")

if NUM_GPU > 1:
    model = multi_gpu_model(model, gpus=NUM_GPU)
opt = optimizers.Adam(lr=INITIAL_LR, decay=OPT_DECAY)
model.compile(loss=l, optimizer=opt)

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

#reduce_rl = ReduceLROnPlateau(monitor='loss', 
#                              factor=0.1,
#                              patience=10,
#                              verbose=0,
#                              mode='auto',
#                              min_delta=0.1,
#                              cooldown=0,
#                              min_lr=0)

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

    #def on_epoch_end(self, epoch, logs=None):
    #    logs.update({'lr': K.eval(self.model.optimizer.lr)})
    #    super().on_epoch_end(epoch, logs)
    
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

checkpoint = ModelCheckpoint(MODEL_SAVE_FILE,
                             monitor='loss',
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

data_gen = data_generator(TRAIN_DIR,
                          ANNOTATIONS_PATH,
                          BATCHSIZE,
                          WIDTH, HEIGHT,
                          anchor_width,
                          anchor_height,
                          num_classes=NUM_CLASSES,
                          sample_type='png')

model.fit_generator(data_gen,
                    steps_per_epoch=steps_epoch,
                    epochs=1000,
                    verbose=1,
                    callbacks=[print_info, tensorboard, checkpoint])

