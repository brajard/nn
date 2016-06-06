#!/usr/bin/env python
### Script to run a complete experiment

## Import libraries
import datatools
reload(datatools)
from datatools import load_dataset, split_base, plot_images, plot_compare
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.objectives import squared_error
import matplotlib.pyplot as plt


### this is really dumb, current nolearn doesnt play well with lasagne,
### so had to manually copy the file I wanted to this folder
from shape import ReshapeLayer

from lasagne.nonlinearities import tanh
import pickle
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score
import os

 

## Load Data
#Load data (by defulat in directory ../data, ubar,vbar)
X = load_dataset(linnum=[134,135])
print 'size of total base',X.shape

nt,nparam,nx,ny = X.shape# nt,2,7,7

mu = np.array([np.mean(X[:,i,:,:].flatten()) for i in range(nparam)])
sigma = np.array([np.std(X[:,i,:,:].flatten()) for i in range(nparam)])

X_train = X.astype(np.float64)
for i in range(nparam):
    X_train[:,i,:,:] = (X_train[:,i,:,:] - mu[i]) / sigma[i]
X_train = X_train.astype(np.float32)
# we need our target to be 1 dimensional
X_out = X_train.reshape((X_train.shape[0], -1))


#Split in learn, test base (using scikit ??)
conv_filters = 16
deconv_filters = 16
filter_sizes = 3
pad = (filter_sizes-1)/2
epochs = 20
encode_size = 5

ae = NeuralNet(
    layers=[
        ('input',layers.InputLayer),
        ('conv', layers.Conv2DLayer),
        ('flatten',ReshapeLayer),
        ('encode_layer',layers.DenseLayer),
        ('hidden',layers.DenseLayer),
        ('unflatten', ReshapeLayer),
        ('deconv', layers.Conv2DLayer),
        ('output_layer',ReshapeLayer),
        ],
    
    input_shape=(None, nparam, nx ,ny),
    conv_num_filters=conv_filters, 
    conv_filter_size = (filter_sizes, filter_sizes),
    conv_pad = (pad,pad),
    conv_nonlinearity=None,
    flatten_shape=(([0], -1)), # not sure if necessary?
    encode_layer_num_units = encode_size,
    hidden_num_units= deconv_filters * (nx + filter_sizes - 1) * (ny + filter_sizes - 1),
    unflatten_shape=(([0], deconv_filters, (nx + filter_sizes - 1), (ny + filter_sizes - 1))),
    deconv_num_filters=nparam, 
    deconv_filter_size = (filter_sizes, filter_sizes),
    deconv_nonlinearity=None,
    #deconv_pad = (pad,pad),
    output_layer_shape = (([0], -1)),
    update_learning_rate = 0.01,
    update_momentum = 0.975,
    regression=True,
    max_epochs= epochs,
    verbose=5,
    )

## Learning
ae.fit(X_train, X_out)

## Test ??
X_predict = ae.predict(X_train)
X_predict = X_predict.reshape(nt,nparam,nx,ny)
plot_compare(X_train,X_predict,[50,1000,5500,7000,11000])
