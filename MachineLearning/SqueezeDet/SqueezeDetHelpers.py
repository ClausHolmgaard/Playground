
# From: https://github.com/omni-us/squeezedet-keras

from keras.layers import Input, Conv2D, concatenate, BatchNormalization, Activation
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
from keras import backend as K
import numpy as np


def fire_layer(name, input, s1x1, e1x1, e3x3, weight_decay, stdd=0.01):
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
        use_bias=False,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation=None,
        kernel_regularizer=l2(weight_decay)
        )(input)

    bn1 = BatchNormalization(name=name+'/bn1')(sq1x1)
    act1 = Activation('relu', name=name+'/act1')(bn1)

    ex1x1 = Conv2D(
        name = name + '/expand1x1',
        filters=e1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation=None,
        kernel_regularizer=l2(weight_decay)
        )(act1)
    
    bn2 = BatchNormalization(name=name+'/bn2')(ex1x1)
    act2 = Activation('relu', name=name+'/act2')(bn2)

    ex3x3 = Conv2D(
        name = name + '/expand3x3',
        filters=e3x3, kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation=None,
        kernel_regularizer=l2(weight_decay)
        )(act2)
    
    bn3 = BatchNormalization(name=name+'/bn3')(ex3x3)
    act3 = Activation('relu', name=name+'/act3')(bn3)

    #return concatenate([ex1x1, ex3x3], axis=3)
    return concatenate([act2, act3], axis=3)

def binary_crossentropy(y, y_hat, epsilon):
    return y * (-np.log(y_hat + epsilon)) + (1-y) * (-np.log(1-y_hat + epsilon))

def keras_binary_crossentropy(y, y_hat, epsilon):
    return y * (-K.log(y_hat + epsilon)) + (1-y) * (-K.log(1-y_hat + epsilon))

if __name__ == "__main__":
    print("test")
    input_layer = Input(shape=(200, 200, 3), name="input")

    fire = fire_layer(name="fire", input = input_layer, s1x1=16, e1x1=64, e3x3=64, weight_decay=0.001)