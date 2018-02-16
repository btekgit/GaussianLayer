#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import keras
from keras.datasets import mnist,fashion_mnist, cifar10
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten,AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from gausslayer import GaussScaler
import numpy as np
from matplotlib import pyplot as plt

tf.reset_default_graph()


batch_size = 128
num_classes = 10
epochs = 10



# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
use_subset=False
if use_subset:
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



node_in  = Input(shape=input_shape)
enable_gauss = True
if enable_gauss:
    
    node_gauss = GaussScaler(rank=2,filters=12,kernel_size=(7,7), 
                          input_shape=input_shape, 
                          padding='same',name='gausslayer')(node_in)
    
    
    #output_at_gauss = node_gauss.
    #node_gauss= AveragePooling2D(pool_size=(2, 2))(node_gauss)

    node_conv1 =Conv2D(30, kernel_size=(3, 3),
                 activation='relu')(node_gauss)#, input_shape=input_shape))

else:
    node_gauss = node_in
    node_conv1 = Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=input_shape)(node_in)

#node_conv1= AveragePooling2D(pool_size=(2, 2))(node_conv1)
node_conv2 = Conv2D(64, (3, 3), activation='relu')(node_conv1)
node_mxpool1 = MaxPooling2D(pool_size=(2, 2))(node_conv2)
node_drp1 =Dropout(0.25)(node_mxpool1)
node_flt= Flatten()(node_drp1)
node_dns1=Dense(128, activation='relu')(node_flt)
node_drp=Dropout(0.5)(node_dns1)
pred=Dense(num_classes, activation='softmax')(node_drp)
model = Model(inputs=node_in, outputs=[pred])
model.summary()

# this is supposed write the training data.
tb_call_back = keras.callbacks.TensorBoard(log_dir='./Graph', 
                                         histogram_freq=1, 
                                         write_graph=False,
                                         write_grads=False,
                                         write_images=False)
#checkpoint = keras.callbacks.ModelCheckpoint('./Model' + '/weights-{epoch:02d}.h5', monitor='accuracy',
#                                       save_best_only=True, save_weights_only=False, verbose=1)

#toptimizer = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
toptimizer = keras.optimizers.Adam(decay=0.001)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=toptimizer,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1230,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

gauss_layer = model.get_layer('gausslayer')    
gauss_layer_var = gauss_layer.get_weights()
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=gauss_layer.output)
print(gauss_layer_var[0][0],gauss_layer_var[0][1],gauss_layer_var[0][11])
intermediate_output = intermediate_layer_model.predict(np.expand_dims(x_test[0], axis=0))
num_filters = intermediate_output.shape[-1]
n_row = int(num_filters/4)
n_col = 4
plt.figure()
plt.subplots(n_row,n_col,figsize=(10,10))
for ix in range(num_filters):
    plt.subplot(n_row,n_col,ix+1)
    plt.imshow(intermediate_output[0,:,:,ix])
plt.show()
