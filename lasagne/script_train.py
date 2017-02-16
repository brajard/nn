#!/usr/bin/env python
### Script to run a complete experiment

## Import libraries
import datatools
import os
from deepnn import kerasnn
from datatools import prepare_data, make_train
import theano

#theano.config.optimizer = 'None'

import numpy as np

import matplotlib.pyplot as plt

plt.close("all")
outdir = '../data/nn_bestnet_dense'

datadir = '/net/argos/data/parvati/aaclod/home2/aaclod/MAREES/python_nn/nn'
fname = 'MATRICE_01_2017.mat'
geofile = 'USABLE_PointbyPoint_01_2017.mat'
data,scaler = prepare_data(datadir=datadir,fname=fname,geofile=geofile,lognorm=True,epsi=0.0001,smNum = [44,55,66,77,88,99,110])

#params = {'n_feat_in_': 5, 'network_type_': 'all', 'n_feat_out_': 7, 'nhid2_': 10, 'nhid1_': 12}
#params = {'network_type_': 'conv', 'n_feat_in_': 12, 'n_feat_out_': 14}
params = {'network_type_': 'dense', 'nhid1_': 6, 'nhid2_': 40}
model = kerasnn(shapef_=data.Xval.shape[1:],nb_epoch_=200)
model.set_params(**params)
make_train(data,model,outdir)
