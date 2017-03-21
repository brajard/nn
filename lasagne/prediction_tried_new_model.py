#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:04:40 2017

@author: cvasseur
"""
#!/usr/bin/env python
import os
from importlib import reload

import xarray as xr
import datatools
reload(datatools)
from datatools_tried import move_data,predict_model
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot

outdir = '../data/new_model'
outdir2 = '../data/prediction_new_model'

data =  'dataval.npz'
histname = 'history.p'
tosave = True
tosavemat = True
tosavemodel = True
plt.close("all")

npzfile = np.load(os.path.join(outdir,data))
Xval = npzfile['Xval']
yval = npzfile['yval']

# prédiction tout d'abord sur les 21 jours avant Xval1 
# puis des 21 jours apres Xval2
Xval1 = Xval[0:504,:]
Xval2 = Xval[504:1008,:]
yval1 =yval[0:504,:]
yval2 =yval[504:1008,:]     


# visualisation des graphes en fonction du nombre de pixels : visualisation
# prediction maximum : max
# nombre d'input ici : t-6, t-5, t-4, t-3, t-2, t-1

#look_back=6 #parametre à modifier dans la fonction prepare data pas ici
max=10 #horizon
nb_model=10

size, lookback, npar, nx ,ny = Xval.shape
prediction_av = np.zeros((504,max,yval.shape[1]))
prediction_ap = np.zeros((504,max,yval.shape[1]))
corr_av = np.zeros(max)
rmse_av = np.zeros(max)
corr_ap = np.zeros(max)
rmse_ap = np.zeros(max)
    

for i in range(0,nb_model):
    modelname = 'rnn_app'+str(i+1)+'.json'
    weights = 'weights_app'+str(i+1)+'.h5'
    
    model = model_from_json(open(os.path.join(outdir,modelname)).read())
    model.load_weights(os.path.join(outdir,weights))
    
    if i != 0:
        Xval1 = move_data(Xval1, prediction_av,i-1)
        Xval2 = move_data(Xval2, prediction_ap,i-1)

    prediction_av[:,i,:], corr_av[i], rmse_av[i] = predict_model(model,Xval1,yval1,i)
    prediction_ap[:,i,:],corr_ap[i], rmse_ap[i] = predict_model(model,Xval2, yval2, i)
    
print(rmse_ap[0])
# save results 
# res des 21 jours avant :
prediction=xr.DataArray(prediction_av)
prediction.name = 'prediction'
corr=xr.DataArray(corr_av)
corr.name='correlation'
rmse=xr.DataArray(rmse_av)
rmse.name = 'rmse'
data=xr.merge([prediction, corr, rmse])
os.makedirs(outdir2,exist_ok=True)
np.savez(os.path.join(outdir2,'prediction21joursav.npz'),prediction=data.prediction,corr=data.correlation, rmse=data.rmse)
#res 21 jours apres
prediction=xr.DataArray(prediction_ap)
prediction.name = 'prediction'
corr=xr.DataArray(corr_ap)
corr.name='correlation'
rmse=xr.DataArray(rmse_ap)
rmse.name = 'rmse'
data=xr.merge([prediction, corr, rmse])
os.makedirs(outdir2,exist_ok=True)
np.savez(os.path.join(outdir2,'prediction21joursap.npz'),prediction=data.prediction,corr=data.correlation, rmse=data.rmse)    