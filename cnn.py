import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as data
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNN:
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, save_path=None):
        (x_train, y_train), (x_test, y_test) = data.mnist.load_data()

        x_train, y_train = augment_data(x_train, y_train)
        x_test, y_test = augment_data(x_test, y_test)

        # normalize images
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=33)
        x_train = tf.expand_dims(x_train, 3)

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.1
        )
        datagen.fit(x_train)

        cnn = make_cnn()
        opt = SGD(lr=0.001, momentum=0.9)
        cnn.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cnn.summary()

        history = cnn.fit(datagen.flow(x_train, y_train, batch_size=100), validation_data=(x_val, y_val), epochs=10)

        score, accuracy = cnn.evaluate(x_test, y_test, verbose=0)
        print('test score: ', score)
        print('accuracy: ', accuracy)

        if save_path is not None:
            cnn.save(save_path)

        self.model = cnn

