import os, csv, keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import initializers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger


def buildModel(mode):

    model = Sequential()

    if mode == "dnn":
        
        model.add(Flatten(input_shape=(48, 48, 1)))

        model.add(Dense(1024))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(1024))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1024))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(7))
        model.add(Activation("softmax"))
        optim = Adam()
    
    if mode == "cnn":
        
        model.add(Convolution2D(64, kernel_size=3, strides=1, padding="same", input_shape=(48, 48, 1), kernel_initializer=initializers.he_normal(seed=None)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.3))

        model.add(Convolution2D(128, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=None)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.3))

        model.add(Convolution2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=None)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.3))

        model.add(Convolution2D(512, kernel_size=3, strides=1, padding="same", kernel_initializer=initializers.he_normal(seed=None)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.4))

        model.add(Flatten())

        model.add(Dense(512, kernel_initializer=initializers.he_normal(seed=None)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(512, kernel_initializer=initializers.he_normal(seed=None)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        
        model.add(Dense(7))
        model.add(Activation("softmax"))
        optim = Adam()

    model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model
