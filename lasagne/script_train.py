#!/usr/bin/env python
### Script to run a complete experiment

## Import libraries
import datatools
import os
from importlib import reload
reload(datatools)
from datatools import plot_images, plot_compare, history2dict, prepare_data
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.pooling import MaxPooling2D
#from keras.layers.extra import TimeDistributedFlatten,TimeDistributedConvolution2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Deconvolution2D,Convolution2D
import theano
import pandas as pd
import pickle

theano.config.optimizer = 'None'

import numpy as np

import matplotlib.pyplot as plt

plt.close("all")

datadir = '/net/argos/data/parvati/aaclod/home2/aaclod/MAREES/python_nn/nn'
fname = 'MATRICE_01_2017.mat'
geofile = 'USABLE_PointbyPoint_01_2017.mat'
data,scaler = prepare_data(datadir=datadir,fname=fname,geofile=geofile,lognorm=False,epsi=0.0)

nt,n_prev,npar,nx,ny = data.Xapp.shape

# Model Definition
in_out_neurons = nx*ny
n_feat = 5
filter_size=3
nhid = 12
pool_size = (2,2)

new_nx = nx//pool_size[0]
new_ny = ny//pool_size[1]

model = Sequential()

model.add(TimeDistributed(Convolution2D(n_feat,filter_size,filter_size,border_mode='same'),input_shape=(n_prev,npar,nx,ny)))
model.add(Activation("linear"))
model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size, strides=None)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(nhid)))
model.add(Activation("relu"))
model.add(LSTM(output_dim=nhid,return_sequences=False))
#model.add(SimpleRNN(input_dim=in_out_neurons,output_dim=nhid,return_sequences=False))
# parent args
#keras.layers.recurrent.Recurrent(weights=None, return_sequences=False, go_backwards=False, stateful=False, unroll=False, consume_less='cpu', input_dim=None, input_length=None)
# heir args
#keras.layers.recurrent.SimpleRNN(output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)

model.add(Dense(input_dim=nhid,output_dim=n_feat*new_nx*new_ny))
model.add(Activation("relu"))
model.add(Reshape((n_feat,new_nx,new_ny)))
model.add(Deconvolution2D(1,filter_size,filter_size,output_shape=(None,1,nx,ny),subsample=pool_size,border_mode='valid'))
model.add(Activation("linear"))
model.add(Flatten())

model.compile(loss="mean_squared_error",optimizer="rmsprop")

history = model.fit(data.Xapp,data.yapp,batch_size=256,nb_epoch=50,validation_split=0.05)
y_predict = model.predict(data.Xapp)

#plot_compare(yapp.reshape([-1,7,7]),y_predict.reshape([-1,7,7]),[50,1000,5500,7000,11000])

# Save nn
outdir = '../data/nn_rec3'
modelname = 'rnn.json'
weights = 'weights.h5'
datatrain =  'datatrain.npz'
dataval = 'dataval.npz'
json_string = model.to_json()

os.makedirs(outdir,exist_ok=True)
pickle.dump(history2dict(history), open(os.path.join(outdir,'history.p'), "wb" ))
open(os.path.join(outdir,modelname),'w').write(json_string)
model.save_weights(os.path.join(outdir,weights),overwrite=True)
np.savez(os.path.join(outdir,datatrain),Xapp=data.Xapp,yapp=data.yapp)
np.savez(os.path.join(outdir,dataval),Xval=data.Xval,yval=data.yval)
