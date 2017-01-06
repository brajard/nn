#!/usr/bin/env python
### Script to run a complete experiment

## Import libraries
import datatools
import os
from importlib import reload
reload(datatools)
from datatools import history2dict, prepare_data, define_model_all
import pickle
import theano

theano.config.optimizer = 'None'

import numpy as np

import matplotlib.pyplot as plt

plt.close("all")

datadir = '/net/argos/data/parvati/aaclod/home2/aaclod/MAREES/python_nn/nn'
fname = 'MATRICE_01_2017.mat'
geofile = 'USABLE_PointbyPoint_01_2017.mat'
data,scaler = prepare_data(datadir=datadir,fname=fname,geofile=geofile,lognorm=True,epsi=0.0001,smNum = [44,55,66,77,88,99,110])


n_feat = 5
filter_size=3
nhid1 = 12
nhid2 = 12
pool_size = (2,2)

model = define_model_all(data.Xapp.shape,n_feat=n_feat,filter_size=filter_size,nhid1=nhid1,nhid2=nhid2,pool_size=pool_size)

history = model.fit(data.Xapp,data.yapp,batch_size=256,nb_epoch=50,validation_split=0.05)
y_predict = model.predict(data.Xapp)

#plot_compare(yapp.reshape([-1,7,7]),y_predict.reshape([-1,7,7]),[50,1000,5500,7000,11000])

# Save nn
outdir = '../data/nn_rec5'
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
