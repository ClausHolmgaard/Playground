
# From: https://github.com/omni-us/squeezedet-keras

from keras.layers import Input, Conv2D, concatenate, BatchNormalization, Activation
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
from keras import backend as K
import numpy as np


def binary_crossentropy(y, y_hat, epsilon):
    return y * (-np.log(y_hat + epsilon)) + (1-y) * (-np.log(1-y_hat + epsilon))

def keras_binary_crossentropy(y, y_hat, epsilon):
    return y * (-K.log(y_hat + epsilon)) + (1-y) * (-K.log(1-y_hat + epsilon))

def get_all_points_from_prediction(pred, anchors, threshold=1.0, offset_weight=1.0, num_classes=1, is_label=True):
    """
    pred is a prediction map in the shape (ANCHOR_HEIGHT, ANCHOR_WIDTH, 3*num_classes)
    """
    # Get all points with a confidence above threshold
    label_indicies = np.where(pred[:, :, 0] >= threshold)
    num_points = len(label_indicies[0])
    points = np.zeros((num_points, 4))
    
    # Loop through all anchor points
    for c, (x_anchor, y_anchor) in enumerate(zip(label_indicies[0], label_indicies[1])):
        # when anchor location is known, the location of the closest anchor in the actual image can be found
        x_without_offset, y_without_offset = anchors[x_anchor, y_anchor]
        
        # The offset can then be extracted from the labels
        (x_offset, y_offset) = pred[label_indicies[0], label_indicies[1]][0][1:]
        if not is_label:
            x_offset = 2 * (x_offset - 0.5)
            y_offset = 2 * (y_offset - 0.5)
        x_offset *= offset_weight
        y_offset *= offset_weight

        points[c] = (x_without_offset, y_without_offset, x_offset, y_offset)
    
    return points

def fire_layer(name, input, s1x1, e1x1, e3x3, stdd=0.01, regularizer=None):
    """
    wrapper for fire layer constructions
    :param name: name for layer
    :param input: previous layer
    :param s1x1: number of filters for squeezing
    :param e1x1: number of filter for expand 1x1
    :param e3x3: number of filter for expand 3x3
    :param stdd: standard deviation used for intialization
    :return: a keras fire layer
    """

    sq1x1 = Conv2D(
        name = name + '/squeeze1x1',
        filters=s1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=True,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation='relu',
        kernel_regularizer=regularizer
        )(input)

    ex1x1 = Conv2D(
        name = name + '/expand1x1',
        filters=e1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=True,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation='relu',
        kernel_regularizer=regularizer
        )(sq1x1)

    ex3x3 = Conv2D(
        name = name + '/expand3x3',
        filters=e3x3, kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=True,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation='relu',
        kernel_regularizer=regularizer
        )(sq1x1)

    return concatenate([ex1x1, ex3x3], axis=3)

if __name__ == "__main__":
    print("test")
    input_layer = Input(shape=(200, 200, 3), name="input")

    fire = fire_layer(name="fire", input = input_layer, s1x1=16, e1x1=64, e3x3=64, weight_decay=0.001)