#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import keras
from keras.datasets import mnist,fashion_mnist, cifar10
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from gausslayer import GaussScaler
import numpy as np

tf.reset_default_graph()


batch_size = 128
num_classes = 10
epochs = 10



# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
debug=True
if debug:
    x_train=x_train[1:2000,]
    y_train=y_train[1:2000,]
# input image dimensions
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
if len(x_train.shape)>3:
    nchannels = x_train.shape[3]
else: 
    nchannels = 1
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], nchannels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], nchannels, img_rows, img_cols)
    input_shape = (nchannels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, nchannels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, nchannels)
    input_shape = (img_rows, img_cols, nchannels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
enable_gauss = True
if enable_gauss:
    model.add(GaussScaler(rank=2,filters=12,kernel_size=(3,3), 
                          input_shape=input_shape, 
                          padding='same',name='input'))

    model.add(Conv2D(30, kernel_size=(3, 3),
                 activation='relu'))#, input_shape=input_shape))

else:
    
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=input_shape))

#model.add(Conv2D(32, kernel_size=(3, 3),
#                 activation='relu'))#, input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

tb_call_back = keras.callbacks.TensorBoard(log_dir='./Graph', 
                                         histogram_freq=1, 
                                         write_graph=True,
                                         write_grads=True,
                                         write_images=True)
#checkpoint = keras.callbacks.ModelCheckpoint('./Model' + '/weights-{epoch:02d}.h5', monitor='accuracy',
#                                       save_best_only=True, save_weights_only=False, verbose=1)


toptimizer = keras.optimizers.Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=toptimizer,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[tb_call_back])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
