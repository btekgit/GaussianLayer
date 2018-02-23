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
from keras.layers import Input, Dense, Dropout, Flatten,AveragePooling2D, Maximum, Subtract, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, MaxPooling3D,SeparableConv2D, Lambda, Concatenate
from keras import backend as K
import tensorflow as tf
from gausslayer_simple import GaussScaler
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python import debug as tf_debug
tf.reset_default_graph()
sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)




batch_size = 128
num_classes = 10
epochs = 10

from datetime import datetime
now = datetime.now()
logdir = "tf_logs/.../" + now.strftime("%Y%m%d-%H%M%S") + "/"

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
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
#node_s = Dropout(0.1)(node_in)
enable_gauss = True
calculate_dog = False
up_down_sample = True
if enable_gauss:
    gauss_filters = 16
    node_gauss = GaussScaler(rank=2,filters=gauss_filters,kernel_size=(9,9), 
                          input_shape=input_shape, 
                          padding='same',name='gausslayer')(node_in)
    

    
    gaussian_channels =[]
    conv_channels =  []
    diff_channels =  []
    for i in range(gauss_filters):
        # separate channels
        g_c= Lambda(lambda x: K.expand_dims(x[:,:,:,i],axis=3), name='ChannelSeparator'+str(i))(node_gauss)
        
        
        # downsample Gaussian layers, however leave the first one as it is. 
        # we want a clear input as well
        # this may be controlled by the cov_scaler value.
        if up_down_sample and i>4:
            g_c = AveragePooling2D(pool_size=(2,2))(g_c)
            
            
        
        g_c_shape = input_shape
        gaussian_channels.append(g_c)
        
        # conv with filter
        node_conv_gc = Conv2D(64, kernel_size=(3, 3), padding='same',
                     activation='relu', input_shape=input_shape)(g_c)
        
        if up_down_sample and i>4:
            node_conv_gc = UpSampling2D(size=(2,2))(node_conv_gc)
        
        conv_channels.append(node_conv_gc)
        
        # this calculates scale-space with DoG
        if i >0 and calculate_dog:
            diff_layer = Subtract()([g_c,gaussian_channels[i-1]])
            diff_channels.append(diff_layer)
        
    
    # depthwise pooling layer
    if not calculate_dog:
        node_conv1 = Maximum(name='merge')(conv_channels)
    else:
        node_conv1 = Concatenate(name='merge',axis=-1)(diff_channels)
    
    
    
    
    
    #output_at_gauss = node_gauss.
    #node_gauss= AveragePooling2D(pool_size=(2, 2))(node_gauss)
    #node_conv1 =K.depthwise_conv2d(30, kernel_size=(3, 3),
    #             activation='relu')
    #node_conv1 =SeparableConv2D(30, kernel_size=(3, 3), depth_multiplier=1,
    #             activation='relu')(node_gauss)#, input_shape=input_shape))

else:
    
    node_conv1 = Conv2D(64, kernel_size=(3, 3),
                     activation='relu', input_shape=input_shape)(node_in)

node_pool1= MaxPooling2D(pool_size=(2, 2))(node_conv1)
#node_pool1= AveragePooling2D(pool_size=(2, 2))(node_conv1)
node_conv2 = Conv2D(64, (3, 3), activation='relu')(node_pool1)
node_mxpool2 = MaxPooling2D(pool_size=(2, 2))(node_conv2)
node_drp1 =Dropout(0.25)(node_mxpool2)

node_flt= Flatten()(node_drp1)
node_dns1=Dense(128, activation='relu')(node_flt)
node_drp=Dropout(0.5)(node_dns1)
pred=Dense(num_classes, activation='softmax')(node_drp)
model = Model(inputs=node_in, outputs=[pred])
model.summary()

# this is supposed write the training data.
tb_call_back = keras.callbacks.TensorBoard(log_dir=logdir, 
                                         histogram_freq=0, 
                                         write_graph=False,
                                         write_grads=False,
                                         write_images=False)

#out_call_back = keras.callbacks.Callback()
#checkpoint = keras.callbacks.ModelCheckpoint('./Model' + '/weights-{epoch:02d}.h5', monitor='accuracy',
#                                       save_best_only=True, save_weights_only=False, verbose=1)

#toptimizer = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
toptimizer = keras.optimizers.Adadelta()
loss_funct = keras.losses.mse # keras.losses.categorical_crossentropy
#toptimizer = keras.optimizers.SGD(lr=0.05)
model.compile(loss=loss_funct,
              optimizer=toptimizer,
              metrics=['accuracy'])
if enable_gauss:
    gauss_layer = model.get_layer('gausslayer')
    U_func = K.function(inputs=[model.input], outputs=[gauss_layer.U()])
    U_val=U_func([np.expand_dims(x_test[0], axis=0)])[0]
    num_filters = U_val.shape[-1]
    n_row = int(num_filters/4)
    n_col = 4

    plt.subplots(n_row,n_col,figsize=(10,10))
    plt.title('Gauss Filters before training')
    for ix in range(num_filters):
        plt.subplot(n_row,n_col,ix+1)
        plt.imshow(U_val[:,:,0,ix])
        plt.colorbar()
    plt.show(block=False)



model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=175,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[tb_call_back])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

if enable_gauss:
    gauss_layer = model.get_layer('gausslayer')    
    gauss_layer_var = gauss_layer.get_weights()
    depthwise_pool = model.get_layer('merge')
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[gauss_layer.output,depthwise_pool.output])
    U_func = K.function(inputs=[model.input], outputs=[gauss_layer.U()])
    U_val=U_func([np.expand_dims(x_test[0], axis=0)])[0]
    for i in range(0,gauss_filters):
        print("Covs: \n", gauss_layer_var[0][i])
        
    intermediate_outputs = intermediate_layer_model.predict(np.expand_dims(x_test[0], axis=0))
    num_filters = intermediate_outputs[0].shape[-1]
    n_row = int(num_filters/4)
    n_col = 4


    plt.subplots(n_row,n_col,figsize=(10,10))
    plt.title('Outputs')
    for ix in range(num_filters):
        plt.subplot(n_row,n_col,ix+1)
        plt.imshow(intermediate_outputs[0][0,:,:,ix])
    plt.show(block=False)

    num_filters = intermediate_outputs[1].shape[-1]
    n_row = int(num_filters/4)
    n_col = 4


    plt.subplots(n_row,n_col,figsize=(12,12))
    plt.title('Pooled outputs')
    
    for ix in range(num_filters):
        plt.subplot(n_row,n_col,ix+1)
        plt.imshow(intermediate_outputs[1][0,:,:,ix])
    plt.show(block=False)
    plt.tight_layout()


    U_func = K.function(inputs=[model.input], outputs=[gauss_layer.U()])
    U_val=U_func([np.expand_dims(x_test[0], axis=0)])[0]
    num_filters = U_val.shape[-1]
    n_row = int(num_filters/4)
    n_col = 4
    plt.subplots(n_row,n_col,figsize=(10,10))
    plt.title('Filters after')
    for ix in range(num_filters):
        plt.subplot(n_row,n_col,ix+1)
        plt.imshow(U_val[:,:,0,ix])
        plt.colorbar()
    plt.show(block=False)


