
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

if __name__ == "__main__":
    print("test")
    input_layer = Input(shape=(200, 200, 3), name="input")

    fire = fire_layer(name="fire", input = input_layer, s1x1=16, e1x1=64, e3x3=64, weight_decay=0.001)