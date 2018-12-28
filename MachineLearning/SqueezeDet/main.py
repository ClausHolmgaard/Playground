import numpy as np
import os
from keras import optimizers
from keras import backend as K
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint

from Models.PoolingAndFire import create_model, create_loss_function
from PreProcess import data_generator, get_num_samples
from GenerateData import generate_data


LOG_DIR = os.path.expanduser("~/logs/SqueezeDet/")
DATA_DIR = os.path.expanduser("~/datasets/Generated")
MODEL_SAVE_FILE = "./results/model_checkpoint.h5py"

BATCHSIZE = 64
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

VALIDATION_SPLIT = 0.3

generate_data(DATA_DIR, WIDTH, HEIGHT, box_min=50, box_max=100, num_images=1000)

model = create_model(320, 320, 3)
out_shape = model.output_shape
anchor_width = out_shape[1]
anchor_height = out_shape[2]
print(f"Needed anchor shape: {anchor_width}x{anchor_height}")

#model.summary()

l = create_loss_function(anchor_width,
                         anchor_height,
                         LABEL_WEIGHT,
                         OFFSET_WEIGHT,
                         OFFSET_LOSS_WEIGHT,
                         EPSILON)

num_samples = get_num_samples(DATA_DIR)
print(f"Number of sampels: {num_samples}")
steps_epoch = num_samples // BATCHSIZE
if steps_epoch < 1:
    steps_epoch = 1
print(f"Steps per epoch: {steps_epoch}")

opt = optimizers.Adam(lr=1e-4, decay=1e-4)
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

print_info = PrintInfo()

tensorboard = TensorBoard(log_dir=LOG_DIR,
                          histogram_freq=0,
                          batch_size=BATCHSIZE,
                          write_graph=True,
                          write_grads=False,
                          write_images=False,
                          embeddings_freq=0,
                          embeddings_layer_names=None,
                          embeddings_metadata=None,
                          embeddings_data=None,
                          update_freq='epoch')

checkpoint = ModelCheckpoint(MODEL_SAVE_FILE,
                             monitor='loss',
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

model.fit_generator(data_generator(DATA_DIR, BATCHSIZE, WIDTH, HEIGHT, anchor_width, anchor_height),
                    steps_per_epoch=steps_epoch,
                    epochs=1000,
                    verbose=1,
                    callbacks=[print_info, tensorboard, checkpoint])

