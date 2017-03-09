#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:38:57 2017

@author: cvasseur
"""

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
from datatools_tried import moy_corr
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
look_back=6 #parametre à modifier dans la fonction prepare data pas ici
visualisation=True
max=200

model = model_from_json(open(os.path.join(outdir,modelname)).read())
model.load_weights(os.path.join(outdir,weights))

if tosavemodel:
    plot(model, show_shapes = True, to_file=os.path.join(outdir,'model.png'))

npzfile = np.load(os.path.join(outdir,data))
Xapp = npzfile['Xval']
yapp = npzfile['yval']
prediction = np.zeros((504,max,49))


# prédiction tout d'abord sur les 21 jours avant Xval1 
# et les 21 jours apres Xval2
Xval1 = Xapp[0:504,:]
Xval2 = Xapp[504:1008,:]
yval1 =yapp[0:504,:]
yval2 =yapp[504:1008,:]

for j in range(0,max-1):
    prediction[:,j,:] = model.predict(Xval1).squeeze()

    # Maj de Xval
    for i in range(1,look_back-1):
        Xval1[:,i-1]=Xval1[:,i]
    Xval1[:,look_back-1]=prediction[:,j,:].reshape([504,1,7,7])
    
    prediction[:,j+1,:] = model.predict(Xval1).squeeze()

# moyenne des corrélations par colonne de la matrice prediction avec les valeurs de yval(ici yapp)
corr = np.zeros(max)
for i in range(0,max):
    corr[i] = moy_corr(yval1,prediction[:,i],i+1)


prediction=xr.DataArray(prediction)
prediction.name = 'prediction'
corr=xr.DataArray(corr)
corr.name='correlation'
data=xr.merge([prediction, corr])

os.makedirs(outdir2,exist_ok=True)
np.savez(os.path.join(outdir2,'prediction21joursav.npz'),prediction=data.prediction,corr=data.correlation)

print('Data saved in directory ../data/prediction/')