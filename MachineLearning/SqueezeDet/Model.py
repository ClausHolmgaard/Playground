
from keras import optimizers
from keras.callbacks import Callback, TensorBoard
from keras.initializers import TruncatedNormal
from datetime import datetime
import os


class CustomModel(object):

    def __init__(self, image_height, image_width, channels, batchsize, log_dir, weight_decay=0.001):

        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.batchsize = batchsize
        self.weight_decay = weight_decay

        self.keep_prob = 0.5
        self.label_weight = 1.0
        self.offset_loss_weight = 1.0
        self.offset_weight = 20.0
        self.epsilon = 1e-16

        self.log_dir = log_dir

        self.model = None
        self.anchor_width = None
        self.anchor_height = None

        self.images = None
        self.labels = None

        self.create_model()
    
    def set_data(self, images, labels):
        self.images = images
        self.labels = labels

    def train(self, optimizer='adam'):
        print_info = PrintInfo()

        if optimizer == 'Adam':
            opt = optimizers.Adam(lr=1e-4, decay=1e-5)
        elif optimizer == 'RMSprop':
            opt = optimizers.RMSprop(lr=0.001,  clipnorm=1.0)
        elif optimizer == 'SGD':
            opt = optimizers.SGD(lr=0.01, decay=0.001, momentum=0.9, nesterov=False)
        elif optimizer == 'Adagrad':
            opt = optimizers.Adagrad(lr=0.001, clipnorm=1.0)

        self.model.compile(loss=self.loss, optimizer=opt)

        timestamp = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        this_log_dir = os.path.join(self.log_dir, timestamp)
        print(f"Logging to {this_log_dir}")

        tb = TensorBoard(log_dir=this_log_dir,
                         histogram_freq=0,
                         batch_size=self.batchsize,
                         write_graph=True,
                         write_grads=False,
                         write_images=False,
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None,
                         embeddings_data=None,
                         update_freq='epoch')

        self.model.fit(self.images.reshape(-1, 320, 320, 3),
                       self.labels.reshape(-1, self.anchor_width, self.anchor_height, 3),
                       batch_size=self.batchsize,
                       epochs=1000,
                       verbose=1,
                       callbacks=[print_info, tb])

class PrintInfo(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(f"Learning rate with decay: {K.eval(lr_with_decay)}")
        print("")