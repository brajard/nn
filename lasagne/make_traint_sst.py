#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:12:04 2017

@author: jbrlod
"""
#%% Init
import xarray as xr
from deepnn import kerasnn
from datatools import  make_train

import os

#%% prepare_data_sst
datadir = '../data'
outdir = '../data/nn_sst'
fname =  'data_sst_keras.nc'

data = xr.open_dataset(os.path.join(datadir,fname))

params = {'n_feat_in_': 5, 'network_type_': 'all', 'n_feat_out_': 7, 'nhid2_': 10, 'nhid1_': 12}
model = kerasnn(shapef_=data.Xval.shape[1:],nb_epoch_=5)
model.set_params(**params)

#%% Train
make_train(data,model,outdir)
