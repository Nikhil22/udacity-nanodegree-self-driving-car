import numpy as np
import tensorflow as tf 

import os
from sklearn.model_selection import train_test_split
import csv

with open('data/driving_log.csv') as f:
    x = [line for line in csv.reader(f)]
x_train, x_val = train_test_split(x, test_size=0.05)

import matplotlib.image as mpimg
from sklearn.utils import shuffle

PREFIX = './data/';

def gen(x, batch):
    while True:
        shuffle(x)
        for offset in range(0, len(x), batch):
            img_list = []
            metrics_list = []
            for xx in x[offset:offset+batch]:
                img_list.append(mpimg.imread(PREFIX + xx[0]))
                metrics_list.append(float(xx[3]))

            yield shuffle(np.array(img_list), np.array(metrics_list))

from keras.layers import Dense, Dropout, Activation, Flatten, ELU
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adam
from keras.layers.core import Lambda


def addElu():
    model.add(ELU())
    
def add_flatten_layer():
    model.add(Flatten())
    
def add_conv_layer(dims, subsample, border_mode):
    model.add(Convolution2D(dims[0], dims[1], dims[2], subsample=subsample, border_mode=border_mode))
    
def conv1():
    add_conv_layer((18,9,9), (4,4), "same")
    addElu()
def conv2():
    add_conv_layer((30,5,5), (2,2), "same")
    addElu()
def conv3():
    add_conv_layer((64,6,6), (2,2), "same")
    add_flatten_layer()
    addElu()
    
def crop():
    model.add(Cropping2D(cropping=((80, 20), (0, 0)),
                     dim_ordering='tf', 
                     input_shape=(160, 320, 3)))
    
def add_lambdas():
    model.add(Lambda(lambda img: tf.image.resize_images(img, [40, 160])))
    model.add(Lambda(lambda x: (x/255.0) - 0.5))

def add_layer(dims, dropout):
    model.add(Dense(dims))
    if dropout:
        model.add(Dropout(dropout))

def build_and_run_model():
    crop()
    add_lambdas()

    # add conv layers
    conv1()
    conv2()
    conv3()

    add_layer(512, 0.5)
    addElu()
    add_layer(50, False)
    addElu()
    add_layer(1, False)
    model.compile(optimizer=Adam(lr=0.0001), loss="mse", metrics=['accuracy'])
    model.summary()

model = Sequential()
build_and_run_model()
    
model.fit_generator(gen(x_train, 32), 
                    samples_per_epoch=len(x_train), 
                    validation_data=gen(x_val, 32),
                    nb_val_samples=len(x_val), nb_epoch=1)
model.save_weights('model.h5');

