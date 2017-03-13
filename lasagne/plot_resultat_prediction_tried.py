#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:40:05 2017

@author: cvasseur
"""
#!/usr/bin/env python
# script affichage des resultats obtenues pour une prédiction jusqu'à t+max
import os
from importlib import reload
import datatools
reload(datatools)
from datatools_tried import plot_horizon, plot_ligne_prediction, plot_temporelle, calcul_persistence
import numpy as np
import matplotlib.pyplot as plt

outdir = '../data/nn_bestnet'
outdir2 = '../data/prediction'
modelname = 'rnn.json'
weights = 'weights.h5'
data =  'dataval.npz'
data2 =  'prediction21joursav.npz'
data3 =  'prediction21joursap.npz'
histname = 'history.p'
tosave = True
tosavemat = True
tosavemodel = True

plt.close("all")

# chargement des predictions obtenues : 21 jours avant
npzfile = np.load(os.path.join(outdir2,data2))
prediction_av = npzfile['prediction']
corr_av = npzfile['corr']

# chargement des predictions obtenues : 21 jours apres
npzfile = np.load(os.path.join(outdir2,data3))
prediction_ap = npzfile['prediction']
corr_ap = npzfile['corr']

npzfile = np.load(os.path.join(outdir,data))
Xapp = npzfile['Xval']
yapp = npzfile['yval']

# étude faite sur les 21 jours avant et 21 jours apres 
Xval1 = Xapp[0:504,:]
Xval2 = Xapp[504:1008,:]
yval1 =yapp[0:504,:]
yval2 =yapp[504:1008,:]

# correlation persistence en fonction de l'horizon
Yval = npzfile['yval']
Ypers=Xapp[:,-1,0].reshape([len(Xapp[:,-1,0]),49])
max=prediction_av[1,:,1].shape[0] #fixe lors de la prediction
corr_hor = calcul_persistence(Ypers,Yval,max)

# -------------------------
# PLT RESULTATS
# ------------------------
look_back=len(Xapp[0,:,0,0,0])
# visualisation des graphes en fonction du nombre de pixels : visualisation
# prediction maximum : max
# nombre d'input ici : t-6, t-5, t-4, t-3, t-2, t-1
# plt correlation en fonction de l'horizon
title='Corrélation en fonction de l horizon (21 jours avant)'
plot_horizon(corr_av,corr_hor,title)
title='Corrélation en fonction de l horizon (21 jours après)'
plot_horizon(corr_ap,corr_hor,title)

#l est la ligne de la matrice que l'on regarde (definie aussi l'horizon dans ce cas)
l=4
title='Évolution d une image sur différents horizons (21 jours avant)'
plot_ligne_prediction(prediction_av,yval1,l,max,title)
title='Évolution d une image sur différents horizons (21 jours après)'
plot_ligne_prediction(prediction_ap,yval2,l,max,title)


# series temporelles des différents horizons (colonne de la matrice)
title='Séries chonologiques 21 jours avant (pixel central)'
horizon=4
plot_temporelle(prediction_av,yval1,horizon,title)
title='Séries chonologiques 21 jours apres (pixel central)'
plot_temporelle(prediction_ap,yval2,horizon,title)
    

plt.show()
        