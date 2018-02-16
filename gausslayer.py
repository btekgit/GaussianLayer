#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:07:34 2018

@author: btek
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras import activations, regularizers, constraints
from keras.engine import InputSpec
import numpy as np
import tensorflow as tf

def idx_init(shape, dtype='float32'):
    idxs = np.zeros((shape[0], shape[1]))
    c = 0
    # assumes square filters
    wid = np.int(np.sqrt(shape[0]))
    hei =np.int(np.sqrt(shape[0]))
    for x in np.arange(wid):  # / (self.incoming_width * 1.0):
        for y in np.arange(hei):  # / (self.incoming_height * 1.0):
            idxs[c, :] = np.array([x, y])
            c += 1

    return idxs

def cov_init(shape, dtype='float32'):
    
    cov = np.identity(shape[1], dtype)
    # shape [0] must have self.incoming_channels * self.num_filters
    cov = np.repeat(cov[np.newaxis], shape[0], axis=0)
    s = np.linspace(0.01, 5, shape[0])
    for t in range(len(s)):
        cov[t] = cov[t] * s[t]
        print(cov[t])
    
    
    return cov

class GaussScaler(Layer):
    def __init__(self, rank, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=False,
                 kernel_regularizer=None,
                 **kwargs):
        super(GaussScaler, self).__init__(**kwargs)
        #def __init__(self, num_filters, kernel_size, incoming_channels=1, **kwargs):
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.input_spec = InputSpec(ndim=self.rank + 2)
                 
        #self.input_shape = input_shape
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'))
        print(kwargs)
        self.kernel_size = kernel_size
        
        self.num_filters = filters
        #self.incoming_channels = incoming_channels
       
        super(GaussScaler, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        
        self.input_channels = input_dim
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        print(kernel_shape)

        self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        # Create a trainable weight variable for this layer.
        
        kernel_size = self.kernel_size
        # Idxs Init
        
        mu = np.array([kernel_size[0] // 2, kernel_size[1] // 2])


        # Convert Types
        self.mu = mu.astype(dtype='float32')

        # Shared Parameters
        # below works for only two dimensional cov 
        self.cov = self.add_weight(shape=[input_dim*self.filters,2,2], 
                                  name="cov", initializer=cov_init, trainable=True,
                                  constraint=constraints.non_neg())
        
        print(self.cov)
        
        # below prepares a meshgrid. 
        self.idxs = self.add_weight(shape=[kernel_size[0]*kernel_size[1],2], 
                                   name="idxs", initializer=idx_init, trainable=False)
        
        super(GaussScaler, self).build(input_shape)  # Be sure to call this somewhere!
        
    
    def U(self):
        
        e1 = (self.idxs - self.mu)
   
        #print(self.cov.shape)
        #print(len(tf.unstack(self.cov,axis=0)))
        #print( tf.linalg.inv(tf.unstack(self.cov,axis=0)[0]))
        # tensorflow does not need scan it does the same op to all covs.
        cov_inv = tf.linalg.inv(self.cov)
        print(cov_inv)
        #cov_inv = K.map_fn(lambda x: tf.linalg.inv(x), elems=tf.unstack(self.cov,axis=0))
       

        e2 = K.dot(e1, K.transpose(cov_inv))
        ex = K.batch_dot(e2, e1, axes=[[1], [1]])
        result = K.exp(-(1 / 2.0) * ex)

   


        # Transpose is super important.
        masks = K.reshape(result,(self.kernel_size[0],
                                  self.kernel_size[1],self.input_channels,self.filters))
        # sum normalization each filter has sum 1
        #sums = tf.reduce_sum(masks, axis=(1, 2), keep_dims=True)
        # Sum normalisation
        #masks = tf.div(masks,sums),

        return masks

    def call(self, inputs):
        
        print(inputs.shape)
        filters = self.U()
        print(filters.shape)
        #filters /= T.sum(filters, axis=(2, 3)).dimshuffle(0, 1, 'x', 'x')
        # channel_first means tensofrlow
        conved = K.depthwise_conv2d(inputs, filters, padding=self.padding, 
                               data_format=self.data_format)
        return conved

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)


#test
def test():
    import tensorflow as tf
    from gausslayer import GaussScaler
    import numpy as np
    
    tf.reset_default_graph()

    from keras.losses import mse
    import keras
    from keras.datasets import mnist,fashion_mnist, cifar10
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Dropout, Flatten
    
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    inputimg = x_train[0]
    
    y = y_train[0]
    sh = (inputimg.shape[0],inputimg.shape[1],1)
    inputs = Input(shape=sh, name='inputlayer')
    node_gauss = GaussScaler(rank=2,filters=12,kernel_size=(7,7), 
                             padding='same',name='gausslayer')(inputs)

    model = Model(inputs=inputs, outputs=node_gauss)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=mse, optimizer=sgd, metrics=['accuracy'])
    model.summary()
    inputimg2 = np.expand_dims(np.expand_dims(inputimg,axis=0), axis=3)
    gauss_layer = model.get_layer('gausslayer')    
    gauss_layer_var = gauss_layer.get_weights()
    out = gauss_layer.get_output_at(0)
    
    scores = model.predict(inputimg2,  verbose=1)
    print( model.get_layer('gausslayer').output )

#with tf.Session() as sess:
#    outval = sess.run(out, feed_dict={inputs:inputimg2})



