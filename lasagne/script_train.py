#!/usr/bin/env python
### Script to run a complete experiment

## Import libraries
import datatools
import os
from importlib import reload
reload(datatools)
from datatools import prepare_data, define_model_all,make_train
import theano

theano.config.optimizer = 'None'

import numpy as np

import matplotlib.pyplot as plt

plt.close("all")
outdir = '../data/nn_rec1'

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


make_train(data,model,outdir,nb_epoch=2)
