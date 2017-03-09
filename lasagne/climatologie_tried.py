#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:13:43 2017

@author: cvasseur
"""

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
outdir = '../data/nn_bestnet'

datadir = '/net/argos/data/parvati/aaclod/home2/aaclod/MAREES/python_nn/nn'
fname = 'MATRICE_01_2017.mat'
geofile = 'USABLE_PointbyPoint_01_2017.mat'
data,scaler = prepare_data(datadir=datadir,fname=fname,geofile=geofile,lognorm=True,epsi=0.0001,smNum = [44,55,66,77,88,99,110], uvNum=[])

# création d'un jour moyen 
Yapp=data.yapp[17:10745,:]
t=np.arange(0,10725,24)
res=np.zeros((24,49))

for i in range(0,24):
    res[i,:]=sum(Yapp[t+i])/447
    
