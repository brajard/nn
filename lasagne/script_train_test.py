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
outdir = '../data/nn_test'


data,scaler = prepare_data(lognorm=True,epsi=0.0001,smNum = [55,77])

params = {'n_feat_in_': 5, 'network_type_': 'all', 'n_feat_out_': 7, 'nhid2_': 10, 'nhid1_': 12}

model = kerasnn(shapef_=data.Xval.shape[1:],nb_epoch_=20)
model.set_params(**params)
make_train(data,model,outdir)
