#!/usr/bin/env python
### Script to run a complete experiment

## Import libraries
import datatools
import os
from importlib import reload
reload(datatools)
from datatools import load_dataset, plot_images, plot_compare, load_sequence, history2dict
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

# Data preparation
n_prev = 6

hor = 1 #prediction horizon. 
#For the time being, hor should be keep to 1

#Epsilon
epsi = 1e-3

#linnum=[44,55,66,77,88,99,110,121,134,135]
linnum = [77,134,135]

Xtmp = load_dataset(linnum=linnum)
#44,55,66,77,88,99,110,121 : TSM at several levels
# 

#first time step to considere
first_time = 8*30*24
Xtsm = np.sum(Xtmp[first_time:,:-2,:,:],axis=1,keepdims=True)
Xtsm[Xtsm<epsi]=epsi
#Xtsm = np.log10(Xtsm+epsi)

Xuv = Xtmp[first_time:,-2:,:,:]

#Normalisation
#Xuv = Xuv.reshape((Xuv.shape[0],Xuv.shape[1],1))
#Xuv = np.mean(Xuv,axis=2)
#mu_uv = np.mean(Xuv,axis=0,keepdims=True)
#sigma_uv = np.std(Xuv,axis=0,keepdims=True)
#Xuv = (Xuv-mu_uv)/sigma_uv


mu_tsm = np.mean(Xtsm.flatten())
sigma_tsm = np.std(Xtsm.flatten())

mu_u = np.mean(Xuv[:,0,:,:].flatten())
mu_v = np.mean(Xuv[:,1,:,:].flatten())
sigma_u = np.std(Xuv[:,0,:,:].flatten())
sigma_v = np.std(Xuv[:,1,:,:].flatten())

Xuv[:,0,:,:] = (Xuv[:,0,:,:]-mu_u)/sigma_u
Xuv[:,1,:,:] = (Xuv[:,1,:,:]-mu_v)/sigma_v
Xtsm = (Xtsm - mu_tsm)/sigma_tsm

X = np.concatenate((Xtsm,Xuv),axis=1)
nt,npar,nx,ny = X.shape


#X = X.reshape([nt,nx*ny])
Xapp,yapp = load_sequence(X,n_prev=n_prev,hor=hor)

yapp = yapp[:,0,:,:]

Xapp = Xapp.reshape([Xapp.shape[0],n_prev,npar,nx,ny])
yapp = yapp.reshape([yapp.shape[0],nx*ny])


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

history = model.fit(Xapp,yapp,batch_size=256,nb_epoch=50,validation_split=0.05)
y_predict = model.predict(Xapp)

#plot_compare(yapp.reshape([-1,7,7]),y_predict.reshape([-1,7,7]),[50,1000,5500,7000,11000])

# Save nn
outdir = '../data/nn_rec1'
modelname = 'rnn.json'
weights = 'weights.h5'
data =  'data.npz'

json_string = model.to_json()

os.makedirs(outdir,exist_ok=True)
pickle.dump(history2dict(history), open(os.path.join(outdir,'history.p'), "wb" ))
open(os.path.join(outdir,modelname),'w').write(json_string)
model.save_weights(os.path.join(outdir,weights),overwrite=True)
np.savez(os.path.join(outdir,data),Xapp=Xapp,yapp=yapp)
