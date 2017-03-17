#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:03:39 2017

@author: cvasseur
"""

#!/usr/bin/env python
### Script to run a complete experiment

## Import libraries
from importlib import reload
import datatools_tried
reload(datatools_tried)
import datatools
reload(datatools)
from deepnn import kerasnn
from datatools_tried import new_data
from datatools import prepare_data, make_train

import numpy as np
import matplotlib.pyplot as plt

plt.close("all")
outdir = '../data/new_model'

datadir = '/net/argos/data/parvati/aaclod/home2/aaclod/MAREES/python_nn/nn'
fname = 'MATRICE_01_2017.mat'
geofile = 'USABLE_PointbyPoint_01_2017.mat'
data,scaler = prepare_data(datadir=datadir,fname=fname,geofile=geofile,lognorm=True,epsi=0.0001,nseq=6,smNum = [44,55,66,77,88,99,110], uvNum=[])

nmodel = 10
# new params
params = {'n_feat_in_': 5, 'network_type_': 'all', 'n_feat_out_': 9, 'nhid2_': 12, 'nhid1_': 12}
# old params
#params = {'n_feat_in_': 5, 'network_type_': 'all', 'n_feat_out_': 7, 'nhid2_': 10, 'nhid1_': 12}

all_model = np.array([kerasnn for _ in range(nmodel)])

for i in range(0,nmodel):
    all_model[i] = kerasnn(shapef_=data.Xval.shape[1:],nb_epoch_=200)
    all_model[i].set_params(**params)
    modelname = 'rnn_app'+str(i+1)+'.json'
    weights = 'weights_app'+str(i+1)+'.h5'
    all_model[i] = make_train(data,all_model[i],outdir,modelname, weights)

    #modified input
    data = new_data(all_model[i], data)