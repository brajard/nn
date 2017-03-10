#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:23:39 2017

@author: cvasseur
"""

#!/usr/bin/env python
import os
from importlib import reload
import datatools
reload(datatools)
from keras.models import model_from_json
from datatools_tried import predict_time
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
max=150

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

look_back=len(Xapp[0,:,0,0,0])

# prediction des 21 jours avants et enregistrement des fichiers
predict_time(model,Xval1,yval1,prediction,max,look_back,outdir2,'prediction21joursav.npz')
predict_time(model,Xval2,yval2,prediction,max,look_back,outdir2,'prediction21joursap.npz')

print('Data saved in directory ../data/prediction/')