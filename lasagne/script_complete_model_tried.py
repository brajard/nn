#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:32:55 2017

@author: cvasseur
"""

## Script complete deep learning model : 1 model

## Import libraries
from importlib import reload
import datatools_tried
reload(datatools_tried)
import datatools
reload(datatools)
import deepnn 
reload(deepnn)
from deepnn import kerasnn
from datatools_tried import new_data, concat_data, update_data
from datatools import prepare_data, make_train

import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np

plt.close("all")
outdir = '../data/complete_model'

datadir = '/net/argos/data/parvati/aaclod/home2/aaclod/MAREES/python_nn/nn'
fname = 'MATRICE_01_2017.mat'
geofile = 'USABLE_PointbyPoint_01_2017.mat'
data,scaler = prepare_data(datadir=datadir,fname=fname,geofile=geofile,lognorm=True,epsi=0.0001,nseq=6,smNum = [44,55,66,77,88,99,110], uvNum=[])

size, lookback, npar, nx ,ny = data.Xapp.shape
    
#nb_it : nomber of iteration to stabilize the model (part 2 of learning)
nb_it = 10
nb_epoch_first = 50
nb_epoch_next = 20
nb_epoch_conv = 5
params = {'n_feat_in_': 5, 'network_type_': 'all', 'n_feat_out_': 9, 'nhid2_': 12, 'nhid1_': 12}

#create new model
cmodel = kerasnn(shapef_=data.Xval.shape[1:],nb_epoch_=nb_epoch_first)
cmodel.set_params(**params)

# PHASE 1 : (3 apprentissages)
print('\n APPRENTISSAGE PHASE 1 : \n')
#first learning
make_train(data,cmodel,outdir)
model=cmodel.nn_

#create new data :
# data : data first learning
# data2 : data second learning
data_t1 = new_data(model,data)
data_all = concat_data(data,data_t1)

#second learning
cmodel.set_params(nb_epoch_=nb_epoch_next)
make_train(data_all,cmodel,outdir)
model=cmodel.nn_

#create new data :
# update data_t1 with new prediction of the model
# concat data and data_t1 modified (updated)
# new_data move data_t1 on the left and add new prediction
# concat data to obtain new learning data
data_t2 = new_data(model,data_t1)
data_t1 = new_data(model,data)
data_tmp = concat_data(data,data_t1)
data_t3 = concat_data(data_tmp,data_t2)

#third learning
cmodel.set_params(nb_epoch_=nb_epoch_next)
make_train(data_t3,cmodel,outdir)
model=cmodel.nn_

prediction_t = model.predict(data_t3.Xapp).squeeze()

print('\n APPRENTISSAGE PHASE 2 : CONVERGENCE ? \n')
cmodel.set_params(nb_epoch_=nb_epoch_conv)
# PHASE 2 : convergence ? 
critere=np.zeros(nb_it)
for i in range(0,nb_it):
    
    data_t2 = new_data(model,data_t1)
    data_t1 = new_data(model,data) 
    data_tmp = concat_data(data,data_t1)
    data_t3 = concat_data(data_tmp,data_t2)

    make_train(data_t3,cmodel,outdir)
    model=cmodel.nn_

    prediction_tplus1 = model.predict(data_t3.Xapp).squeeze()
    
    # difference between 2 iterations  
    critere[i]=LA.norm(prediction_t-prediction_tplus1)
    
    prediction_t = prediction_tplus1


plt.figure()
plt.plot(critere)
plt.xlabel('itération')
plt.ylabel('écart entre deux itérations')
plt.show()

