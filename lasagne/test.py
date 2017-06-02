#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:52:41 2017

@author: jbrlod
"""

#%% init
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from keras.models import Sequential, Model
from keras.layers import Input 
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.pooling import MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Cropping2D,Deconvolution2D,Conv2DTranspose,Conv2D,UpSampling2D
#%% input param
n_feat_in = (5,10,15)
n_feat_out = n_feat_in[::-1] #reverssed
filter_size_in = (7,5,3)
filter_size_out = filter_size_in[::-1]
pool_size = (2,2)
shape = (36080,6,1,31,31)

#%% def function

def cropsize (insize,outsize):
    """Define the crop size for making insize to be outsize"""
    f=lambda delta:(delta//2,delta-delta//2)
    return tuple(f(ins-out) for (ins,out) in zip(insize,outsize))
    

def define_model_cnn(shape,
                     n_feat_in,
                     n_feat_out,
                     filter_size_in,
                     filter_size_out,
                     pool_size):
    Lshape = [shape]
   
    input_img = Input(shape=shape[1:])
    x = input_img
    print(Model(inputs=input_img,outputs=x).output_shape)   
    lsize = [] #succesive size to reconstruct
    first = True #Special treatment for the first iteration
    for (nfeat,filtsize) in zip(n_feat_in,filter_size_in):
        lsize.append(Model(inputs=input_img,outputs=x).output_shape[-2:])
        x = TimeDistributed(Conv2D(nfeat,(filtsize,filtsize),
                                   data_format='channels_first',
                                   padding='same'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=pool_size,
                                         data_format='channels_first'))(x)
        print(Model(inputs=input_img,outputs=x).output_shape)   
    
    oshape = Model(inputs=input_img,outputs=x).output_shape
    x = TimeDistributed(Flatten())(x)
    print(lsize)
    print(Model(inputs=input_img,outputs=x).output_shape)
     
    #TODO : LSTM , Deconv
    nbatch,nseq,nfeat = Model(inputs=input_img,outputs=x).output_shape
    
#    
    x = LSTM(int(nfeat),return_sequences=False)(x)
    x = Reshape(oshape[2:])(x)#same shape of the output of the encoder
    #without batch and nseq
    print(Model(inputs=input_img,outputs=x).output_shape)
    
    first = True #Special treatment for the first iteration

    for (nfeat,filtsize) in zip(n_feat_out,filter_size_out):
        
        x = Conv2DTranspose(nfeat,(filtsize,filtsize),
                   data_format='channels_first',
                   padding='valid')(x)
        x= UpSampling2D(size=pool_size,
                            data_format='channels_first')(x)

        cropping = cropsize(Model(inputs=input_img,outputs=x).output_shape[-2:],lsize.pop())
        x=Cropping2D(cropping=cropping, data_format='channels_first')(x)
        print(Model(inputs=input_img,outputs=x).output_shape)
#    
    model = Model (inputs=input_img,outputs=x)         
    return x,model
#%%execute function
x,model = define_model_cnn(shape,n_feat_in,n_feat_out,filter_size_in,filter_size_out,pool_size)
