#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:23:39 2017

@author: cvasseur
"""

#!/usr/bin/env python
import os
from importlib import reload

import xarray as xr
import datatools
reload(datatools)
from datatools import denormalize
from datatools_tried import moy_corr, predict_time, jointPlot
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot

outdir = '../data/nn_bestnet'
outdir2 = '../data/prediction'
modelname = 'rnn.json'
weights = 'weights.h5'
data =  'dataval.npz'
histname = 'history.p'
tosave = True
tosavemat = True
tosavemodel = True
plt.close("all")


# visualisation des graphes en fonction du nombre de pixels : visualisation
# prediction maximum : max
# nombre d'input ici : t-6, t-5, t-4, t-3, t-2, t-1

#look_back=6 #parametre à modifier dans la fonction prepare data pas ici
max=10 #horizon


model = model_from_json(open(os.path.join(outdir,modelname)).read())
model.load_weights(os.path.join(outdir,weights))

if tosavemodel:
    plot(model, show_shapes = True, to_file=os.path.join(outdir,'model.png'))

npzfile = np.load(os.path.join(outdir,data))
Xapp = npzfile['Xval']
yapp = npzfile['yval']


# prédiction tout d'abord sur les 21 jours avant Xval1 
# et les 21 jours apres Xval2
Xval1 = Xapp[0:504,:]
Xval2 = Xapp[504:1008,:]
yval1 =yapp[0:504,:]
yval2 =yapp[504:1008,:]

look_back=len(Xapp[0,:,0,0,0])

# prediction des 21 jours avants et enregistrement des fichiers
corr_av,rmse_av, prediction_av = predict_time(model,Xval1,yval1,max,outdir2,'prediction21joursav.npz')
corr_av =np.asarray(corr_av)
rmse_av = np.asarray(rmse_av)

print('Corralation des 21 jours avant aux horizons 1, 2 et 4 :\n', corr_av[0],' | ',corr_av[1],' | ',corr_av[3])
print('RMSE des 21 jours avant aux horizons 1, 2 et 4 :\n', rmse_av[0],' | ',rmse_av[1],' | ',rmse_av[3])

corr_ap, rmse_ap, prediction_ap = predict_time(model,Xval2,yval2,max,outdir2,'prediction21joursap.npz')
corr_ap =np.asarray(corr_ap)
rmse_ap = np.asarray(rmse_ap)

fname='../data/LSTM/imgplot_pred_hor1.png'
title='Le 9 décembre 2008 entre 12h et 13h (horizon 1)'
jointPlot(np.asarray(prediction_ap[6,0,:]).reshape([7,7]),fname,title)
fname='../data/LSTM/imgplot_truth.png'
title='Le 9 décembre 2008 entre 12h et 13h (truth)'
jointPlot(yval2[6].reshape([7,7]),fname,title)
fname='../data/LSTM/imgplot_pred_hor6.png'
title='Le 9 décembre 2008 entre 12h et 13h (horizon 6)'
jointPlot(np.asarray(prediction_ap[0,6,:]).reshape([7,7]),fname,title)

print('Corralation des 21 jours apres aux horizons 1, 2 et 4 :\n', corr_ap[0],' | ',corr_ap[1],' | ',corr_ap[3])
print('RMSE des 21 jours apres aux horizons 1, 2 et 4 :\n', rmse_ap[0],' | ',rmse_ap[1],' | ',rmse_ap[3])
print('-----------------------------------------------------------------')

print('Data saved in directory ../data/prediction/')
