from keras.layers import *
from SqueezeDetHelpers import *
from keras.models import Model
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
    
    def create_model(self):
        input_layer = Input(shape=(self.image_height, self.image_width, self.channels), name="input")

        conv1 = Conv2D(name='conv1', filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="SAME",
                    kernel_initializer=TruncatedNormal(stddev=0.01),
                    )(input_layer)

        pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool1")(conv1)
        
        fire1 = fire_layer(name="fire1", input=pool1, s1x1=32, e1x1=128, e3x3=128, weight_decay=self.weight_decay)
        fire2 = fire_layer(name="fire2", input=fire1, s1x1=32, e1x1=128, e3x3=128, weight_decay=self.weight_decay)
        
        pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool2")(fire2)
        
        fire3 = fire_layer(name="fire3", input=pool2, s1x1=48, e1x1=192, e3x3=192, weight_decay=self.weight_decay)
        fire4 = fire_layer(name="fire4", input=fire3, s1x1=48, e1x1=192, e3x3=192, weight_decay=self.weight_decay)
        
        pool3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool3")(fire4)

        pred_conf = Conv2D(name='pred_conf', filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding="SAME",
                kernel_initializer=TruncatedNormal(stddev=0.01),
                )(pool3)

        pred_offset = Conv2D(name='pred_offset', filters=2, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding="SAME",
                    kernel_initializer=TruncatedNormal(stddev=0.01),
                    )(pool3)

        preds = concatenate([pred_conf, pred_offset])

        self.anchor_width = int(preds.shape[1])
        self.anchor_height = int(preds.shape[2])
        
        print(f"Output shape: {preds.shape}")

        self.model = Model(inputs=input_layer, outputs=preds)

    def loss(self, y_true, y_pred):
        # We are predicting a batchsize x anchorwidth x anchorheight x 3 output.
        c_predictions = y_pred[:, :, :, 0]
        c_labels = y_true[:, :, :, 0]
        
        pred_offset_x = 2 * (y_pred[:, :, :, 1] - 0.5) * self.offset_weight
        pred_offset_y = 2 * (y_pred[:, :, :, 2] - 0.5) * self.offset_weight
        
        true_offset_x = y_true[:, :, :, 1]
        true_offset_y = y_true[:, :, :, 2]
        
        g_x = K.less(true_offset_x, 0)
        l_x = K.greater(true_offset_x, 0)
        g_y = K.greater(true_offset_y, 0)
        l_y = K.less(true_offset_y, 0)
        
        g_x_i = K.cast(g_x, dtype='float32')
        l_x_i = K.cast(l_x, dtype='float32')
        g_y_i = K.cast(g_y, dtype='float32')
        l_y_i = K.cast(l_y, dtype='float32')

        mask_offset_x = K.clip(g_x_i + l_x_i, 0, 1.0)
        mask_offset_y = K.clip(g_y_i + l_y_i, 0, 1.0)
        
        # number of labels
        num_labels = K.sum(c_labels)
        num_non_labels = self.anchor_width * self.anchor_height - num_labels
        
        # Loss matrix for all entries
        loss_m_all = keras_binary_crossentropy(c_labels, c_predictions, self.epsilon)
        
        # Loss matrix for the correct label
        loss_m_label = keras_binary_crossentropy(c_labels, c_predictions, self.epsilon) * c_labels
        
        # Loss matrix for non labels
        loss_m_nonlabel = loss_m_all - loss_m_label
        
        # Summing and adding weight to label loss
        c_loss_label = K.sum(
            loss_m_label
        ) / self.batchsize / num_labels
        
        # summing and adding weight to non label loss
        c_loss_nonlabel = K.sum(
            loss_m_nonlabel
        ) / self.batchsize / num_non_labels
        
        c_loss = c_loss_label * self.label_weight + c_loss_nonlabel * (1 / self.label_weight)
        
        o_loss_x = K.sum(
            K.square((true_offset_x - pred_offset_x) * mask_offset_x)
        ) / self.batchsize / num_labels
        
        o_loss_y = K.sum(
            K.square((true_offset_y - pred_offset_y) * mask_offset_y)
        ) / self.batchsize / num_labels
        
        o_loss = (o_loss_x + o_loss_y) * self.offset_loss_weight
        
        total_loss = o_loss + c_loss
        
        return total_loss
    
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