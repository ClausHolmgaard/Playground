from keras.layers import *
from SqueezeDetHelpers import *
from keras.models import Model
from SqueezeDetHelpers import *


def create_model(width, height, channels, weight_decay=0):
    input_layer = Input(shape=(width, height, channels), name="input")

    conv1 = Conv2D(name='conv1', filters=128, kernel_size=(3, 3), strides=(2, 2), activation=None, padding="SAME",
                use_bias=False,
                kernel_initializer=TruncatedNormal(stddev=0.01),
                kernel_regularizer=l2(weight_decay)
                )(input_layer)

    bn = BatchNormalization(name='bn')(conv1)
    act = Activation('relu', name='act')(bn)

    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool1")(act)

    fire1_1 = fire_layer(name="fire1_1", input=pool1, s1x1=32, e1x1=128, e3x3=128, weight_decay=weight_decay)
    fire1_2 = fire_layer(name="fire1_2", input=fire1_1, s1x1=32, e1x1=128, e3x3=128, weight_decay=weight_decay)

    pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool2")(fire1_2)

    fire2_1 = fire_layer(name="fire2_1", input=pool2, s1x1=48, e1x1=192, e3x3=192, weight_decay=weight_decay)
    fire2_2 = fire_layer(name="fire2_2", input=fire2_1, s1x1=48, e1x1=192, e3x3=192, weight_decay=weight_decay)

    pool3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool3")(fire2_2)

    fire3_1 = fire_layer(name="fire3_1", input=pool3, s1x1=48, e1x1=192, e3x3=192, weight_decay=weight_decay)
    fire3_2 = fire_layer(name="fire3_2", input=fire3_1, s1x1=48, e1x1=192, e3x3=192, weight_decay=weight_decay)

    pred_conf = Conv2D(name='pred_conf', filters=1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding="SAME",
                kernel_initializer=TruncatedNormal(stddev=0.01),
                
                )(fire3_2)

    pred_offset = Conv2D(name='pred_offset', filters=2, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding="SAME",
                kernel_initializer=TruncatedNormal(stddev=0.01)
                )(fire3_2)

    preds = concatenate([pred_conf, pred_offset])

    return Model(inputs=input_layer, outputs=preds)

def create_loss_function(anchor_width, anchor_height, label_weight, offset_weight, offset_loss_weight, epsilon, batchsize):

    def loss_function(y_true, y_pred):
        # We are predicting a batchsize x anchorwidth x anchorheight x 3 output.
        c_predictions = y_pred[:, :, :, 0]
        c_labels = y_true[:, :, :, 0]
        
        pred_offset_x = 2 * (y_pred[:, :, :, 1] - 0.5) * offset_weight
        pred_offset_y = 2 * (y_pred[:, :, :, 2] - 0.5) * offset_weight
        
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
        num_labels = K.sum(c_labels) + 1
        num_non_labels = anchor_width * anchor_height - num_labels
        
        # Loss matrix for all entries
        loss_m_all = keras_binary_crossentropy(c_labels, c_predictions, epsilon)
        
        # Loss matrix for the correct label
        loss_m_label = keras_binary_crossentropy(c_labels, c_predictions, epsilon) * c_labels
        
        # Loss matrix for non labels
        loss_m_nonlabel = loss_m_all - loss_m_label
        
        # Summing and adding weight to label loss
        c_loss_label = K.sum(
            loss_m_label
        ) / num_labels
        
        # summing and adding weight to non label loss
        c_loss_nonlabel = K.sum(
            loss_m_nonlabel
        ) / num_non_labels
        
        c_loss = c_loss_label * label_weight + c_loss_nonlabel * (1 / label_weight)
        
        o_loss_x = K.sum(
            K.square((true_offset_x - pred_offset_x) * mask_offset_x)
        ) / num_labels
        
        o_loss_y = K.sum(
            K.square((true_offset_y - pred_offset_y) * mask_offset_y)
        ) / num_labels
        
        o_loss = (o_loss_x + o_loss_y) * offset_loss_weight
        
        total_loss = (o_loss + c_loss)
        
        return total_loss

    return loss_function