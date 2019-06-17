#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:07:34 2018
LAst update Jun 17 2019

@author: btek
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras import activations, regularizers, constraints
from keras import initializers
from keras.engine import InputSpec
import numpy as np
import tensorflow as tf

def idx_init(shape, dtype='float32'):
    idxs = np.zeros((shape[0], shape[1]),dtype)
    c = 0
    # assumes square filters
    print("Hereeee")
    wid = np.int(np.sqrt(shape[0]))
    hei =np.int(np.sqrt(shape[0]))
    f = np.float32
    for x in np.arange(wid):  # / (self.incoming_width * 1.0):
        for y in np.arange(hei):  # / (self.incoming_height * 1.0):
            idxs[c, :] = np.array([x/f(wid), y/f(hei)],dtype)
            c += 1

    return idxs

def cov_init(shape, dtype='float32'):
    
    cov = np.identity(shape[1], dtype)
    # shape [0] must have self.incoming_channels * self.num_filters
    cov = np.repeat(cov[np.newaxis], shape[0], axis=0)
    
    #for t in range(shape[0]):
    #    cov[t] = cov[t]
    return cov

def scale_init(shape, dtype='float32'):
    #sc = np.linspace(0.5, 1.6, shape[0]) #best for mnist cluttered
    #sc = np.linspace(0.05, 0.1, shape[0],dtype=dtype) #best for mnist cluttered
    #sc = 0.05*np.ones(shape[0],dtype=dtype) #best for mnist cluttered
    sc = np.linspace(0.1, 0.1, shape[0],dtype=dtype)#tried on fashion mnist with no difference
    #sc=np.expand_dims(sc, axis=1)
    #sc=np.expand_dims(sc, axis=2)
    #print(sc)
    return sc

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
                 gain=1.0,
                 output_padding=None,
                 **kwargs):
        super(GaussScaler, self).__init__(**kwargs)
        #def __init__(self, num_filters, kernel_size, incoming_channels=1, **kwargs):
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = data_format
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.gain = gain
                 
        #self.input_shape = input_shape
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'))
        print(kwargs)
        self.kernel_size = kernel_size
        
        self.num_filters = filters
        #self.incoming_channels = incoming_channels
        print("HEREERREEE 1")
        
        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                     'greater than output padding ' +str(self.output_padding))
                    
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
        print("kernel shape:",kernel_shape)

        self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        # Create a trainable weight variable for this layer.
        
        kernel_size = self.kernel_size
        # Idxs Init
        
        
        
        
        #mu = np.array([kernel_size[0] // 2, kernel_size[1] // 2])
        mu = np.array([0.5, 0.5])


        # Convert Types
        self.mu = mu.astype(dtype='float32')

        # Shared Parameters
        # below works for only two dimensional cov 
        #self.cov = self.add_weight(shape=[input_dim*self.filters,2,2], 
        #                          name="cov", initializer=cov_init, trainable=False)
        
        
        self.cov_scaler = self.add_weight(shape=(self.filters,),
                                          name='scaler',initializer=scale_init,
                                          trainable=True,
                                          constraint= constraints.NonNeg())
                                  #constraint=constraints.non_neg())
        
        #print("Self.cov:",self.cov)
        #print("Self cov-scaler",self.cov_scaler)
        
        # below prepares a meshgrid. 
        #self.idxs = self.add_weight(shape=[kernel_size[0]*kernel_size[1],2], 
        #                           name="idxs", initializer=idx_init, trainable=False)
        
        self.idxs= idx_init(shape=[kernel_size[0]*kernel_size[1],2])
        
        super(GaussScaler, self).build(input_shape)  # Be sure to call this somewhere!
        
    
    def U(self):
  
        e1 = (self.idxs - self.mu)
        print("e1.shape",e1.shape)
        print("cov scaler shape",self.cov_scaler)
   
        #print(self.cov.shape)
        #print(len(tf.unstack(self.cov,axis=0)))
        #print( tf.linalg.inv(tf.unstack(self.cov,axis=0)[0]))
        # tensorflow does not need scan it does the same op to all covs.
        #cov_inv = self.cov
        #cov_scaled =self.cov_scaler*self.cov
#        cov_scaled = tf.scalar_mul(self.cov_scaler,self.cov)
#        print(self.cov.shape, self.cov_scaler.shape )
#        cov_scaled = K.batch_dot(self.cov_scaler,self.cov, axes=[1,2])
        #cov_inv = tf.linalg.inv(cov_scaled)
        #print("cov_scaled :",cov_scaled.shape)
        #cov_inv = K.map_fn(lambda x: tf.linalg.inv(x), elems=tf.unstack(self.cov,axis=0))
       

        #e2 = K.dot(e1, K.transpose(cov_inv))
        #ex = K.batch_dot(e2, e1, axes=[[1], [1]])
        #result = K.exp(-(1 / 2.0) * ex)

        up= K.sum((self.idxs - self.mu)**2, axis=1)
        print("up.shape",up.shape)
        up = K.expand_dims(up,axis=1,)
        print("up.shape",up.shape)
        # clipping scaler in range to prevent div by 0 or negative cov. 
        cov_scaler = K.clip(self.cov_scaler,0.01,5)
        #cov_scaler = self.cov_scaler
        dwn = 2 * (cov_scaler ** 2)
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        result = K.exp(-up / dwn)
        


        # Transpose is super important.
        #filter: A 4-D `Tensor` with the same type as `value` and shape
        #`[height, width, output_channels, in_channels]`
        # we do not care about input channels
        
        masks = K.reshape(result,(self.kernel_size[0],
                                  self.kernel_size[1],
                                  self.filters,1))   
            
        #sum normalization each filter has sum 1
        #sums = K.sum(masks**2, axis=(0, 1), keepdims=True)
        #print(sums)
        gain = K.constant(self.gain, dtype='float32')
        #masks /= K.sqrt(K.sum(masks**2, axis=(0, 1),keepdims=True))
        #masks /= K.sum(masks, axis=(0, 1),keepdims=True)
        masks /= (self.kernel_size[0]*self.kernel_size[1])
        
        #masks *= (gain*np.sqrt(self.kernel_size[0]*self.kernel_size[1]))
        #ums = sums * sums
        #print("sums shape: ", sums.shape)
        
        # Sum normalisation
        
        #masks = masks * (gain/K.sqrt(sums))
        #masks = masks * (gain/sums)
        #print("masks shape", masks.shape)
        #print("masks mask", K.mean(masks))
        return masks

# =============================================================================
#     def call(self, inputs):
#         
#         print(inputs.shape)
#         filters = self.U()
#         print(filters.shape)
#         #filters /= T.sum(filters, axis=(2, 3)).dimshuffle(0, 1, 'x', 'x')
#         # channel_first means tensofrlow
#         conved = K.conv2d(inputs, filters, padding=self.padding, strides=self.strides,
#                                data_format=self.data_format)
#         return conved
# =============================================================================
        

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
          h_axis, w_axis = 2, 3
          c_axis= 1
          
        else:
            h_axis, w_axis = 1, 2
            c_axis=3
            
        ##BTEK 
        kernel = self.U()
        in_channels =input_shape[c_axis]
        
        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_length(height,
                                              stride_h, kernel_h,
                                              self.padding,
                                              out_pad_h,
                                              self.dilation_rate[0])
        out_width = conv_utils.deconv_length(width,
                                             stride_w, kernel_w,
                                             self.padding,
                                             out_pad_w,
                                             self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        ##BTEK 
        kernel = self.U()
        print("kernel shape in output:",kernel.shape)
        print("channel axis")
        kernel = K.repeat_elements(kernel, self.input_channels, axis=c_axis)
        print("kernel reshaped :",kernel.shape)
        outputs = K.conv2d_transpose(
            inputs,
            kernel,
            output_shape,
            self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

# =============================================================================
#     def compute_output_shape(self, input_shape):
#         print("shapeeee")
#         if self.data_format == 'channels_last':
#             space = input_shape[1:-1]
#             new_space = []
#             for i in range(len(space)):
#                 new_dim = conv_utils.conv_output_length(
#                     space[i],
#                     self.kernel_size[i],
#                     padding=self.padding,
#                     stride=self.strides[i],
#                     dilation=self.dilation_rate[i])
#                 new_space.append(new_dim)
#             return (input_shape[0],) + tuple(new_space) + (self.filters,)
#         if self.data_format == 'channels_first':
#             space = input_shape[2:]
#             new_space = []
#             for i in range(len(space)):
#                 new_dim = conv_utils.conv_output_length(
#                     space[i],
#                     self.kernel_size[i],
#                     padding=self.padding,
#                     stride=self.strides[i],
#                     dilation=self.dilation_rate[i])
#                 new_space.append(new_dim)
#             return (input_shape[0], self.filters) + tuple(new_space)
# =============================================================================
        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_length(output_shape[h_axis],
                                                        stride_h,
                                                        kernel_h,
                                                        self.padding,
                                                        out_pad_h,
                                                        self.dilation_rate[0])
        output_shape[w_axis] = conv_utils.deconv_length(output_shape[w_axis],
                                                        stride_w,
                                                        kernel_w,
                                                        self.padding,
                                                        out_pad_w,
                                                        self.dilation_rate[1])
        return tuple(output_shape)





    
    
#test
def test():
    import tensorflow as tf
    #from gausslayer import GaussScaler
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
                             data_format='channels_last',
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

#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        outval = sess.run(out, feed_dict={inputs:inputimg2})
        
#test()
        
def test_mnist():
    #from __future__ import print_function
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import BatchNormalization
    from keras.optimizers import SGD
    from keras import backend as K
    
    batch_size = 128
    num_classes = 10
    epochs = 1
    
    # input image dimensions
    img_rows, img_cols = 28, 28
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
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
    #=============================================================================
    model.add(GaussScaler(rank=2,filters=4,kernel_size=(5,5), 
                           data_format='channels_last',strides=1,
                           padding='same',name='gausslayer', activation='linear',
                           input_shape=input_shape))
    model.add(BatchNormalization())
    #=============================================================================
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
 
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='softmax'))
    
    
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    from lr_multiplier import LearningRateMultiplier
    
    multipliers = {'gausslayer': 0.0}
    opt = LearningRateMultiplier(SGD, lr_multipliers=multipliers, 
                                 lr=0.01, momentum=0.9,decay=0.00)
    print(opt)
    #opt = SGD(lr=0.01,momentum=0.5)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    
    plt = True
    if plt:
        print("Plotting kernels before...")
        import matplotlib.pyplot as plt
        gauss_layer = model.get_layer('gausslayer')
        ws = gauss_layer.get_weights()
        print("Sigmas before",ws[0])
        u_func = K.function(inputs=[model.input], outputs=[gauss_layer.U()])
        output_func = K.function(inputs=[model.input], outputs=[gauss_layer.output])
    
        U_val=u_func([np.expand_dims(x_test[0], axis=0)])
        
        print("U shape", U_val[0].shape)
        print("U max:", np.max(U_val[0][:,:,:,:]))
        num_filt=min(U_val[0].shape[2],12)
        fig=plt.figure(figsize=(10,5))
        for i in range(num_filt):
            ax1=plt.subplot(1, num_filt, i+1)
            im = ax1.imshow(np.squeeze(U_val[0][:,:,i,0]))
        fig.colorbar(im, ax=ax1)
        plt.show()
        
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    

    if plt:
        print("Plotting kernels after ...")
        
        print("U max:", np.max(U_val[0][:,:,:,:]))
        import matplotlib.pyplot as plt
        ws = gauss_layer.get_weights()
        print("Sigmas before",ws[0])
        U_val=u_func([np.expand_dims(x_test[2], axis=0)])
        
        print("U shape", U_val[0].shape)
        num_filt=min(U_val[0].shape[2],12)
        fig=plt.figure(figsize=(16,5))
        for i in range(num_filt):
            ax=plt.subplot(1, num_filt, i+1)
            im = ax.imshow(np.squeeze(U_val[0][:,:,i,0]))
        fig.colorbar(im, ax=ax1)
        plt.show()
        
        
        print("outputs  ...")
        
        n=5
        
        out_val=output_func([np.expand_dims(x_test[5], axis=0)])
        print("Outputs shape", out_val[0].shape)
        num_filt=min(out_val[0].shape[3],12)
        fig=plt.figure(figsize=(16,10))
        ax=plt.subplot(1, num_filt+1, 1)
        im = ax.imshow(np.squeeze(x_test[5]))
        for i in range(num_filt):
            ax=plt.subplot(1, num_filt+1, i+2)
            im = ax.imshow(np.squeeze(out_val[0][0,:,:,i]))
        fig.colorbar(im, ax=ax)
        plt.show()
        
        print("input mean,var,max",np.mean(x_test[5]),np.var(x_test[5]),np.max(x_test[5]))
        print("ouput mean,var,max",np.mean(out_val[0][0,:,:,i]),
                                           np.var(out_val[0][0,:,:,i]),
                                           np.max(out_val[0][0,:,:,i]))
    
    

test_mnist()



