import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import optimizers

def create_model():
    inp = Input(shape=(1,), name="input")
    h = Dense(units=1, activation='linear')(inp)
    model = Model(inputs=inp, outputs=h)

    opt = optimizers.Adam(lr=1e-1)
    model.compile(loss='mse', optimizer=opt)

    return model

