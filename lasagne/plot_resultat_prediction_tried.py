#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:40:05 2017

@author: cvasseur
"""


#!/usr/bin/env python
import os
from importlib import reload
import datatools
reload(datatools)
from datatools_tried import plot_horizon
import numpy as np
import matplotlib.pyplot as plt

outdir = '../data/nn_bestnet'
outdir2 = '../data/prediction'
modelname = 'rnn.json'
weights = 'weights.h5'
data =  'dataval.npz'
data2 =  'prediction21joursav.npz'
histname = 'history.p'
tosave = True
tosavemat = True
tosavemodel = True

plt.close("all")

npzfile = np.load(os.path.join(outdir,data))
Xapp = npzfile['Xval']
yapp = npzfile['yval']

Xval1 = Xapp[0:504,:]
Xval2 = Xapp[504:1008,:]
yval1 =yapp[0:504,:]
yval2 =yapp[504:1008,:]

npzfile = np.load(os.path.join(outdir2,data2))
prediction = npzfile['prediction']
corr = npzfile['corr']



# PARAMETRES
# visualisation des graphes en fonction du nombre de pixels : visualisation
# prediction maximum : max
# nombre d'input ici : t-6, t-5, t-4, t-3, t-2, t-1
look_back=6 #parametre à modifier dans la fonction prepare data pas ici
visualisation=True
max=prediction[1,:,1].shape[0] #fixe lors de la prediction


# plt correlation en fonction de l'horizon
plot_horizon(corr)

#l est la ligne de la matrice que l'on regarde (definie aussi l'horizon dans ce cas)
l=6
string = []
if l > max :
    plt.figure()
    for i in range(0,max):
        plt.plot(prediction[l-i-1,i,:])
        string.append('horizon ' + str(i+1))
    plt.plot(yval1[l-1])
    string.append('truth')
    plt.legend(string)
    plt.xlabel('pixels')
    plt.ylabel('quantité de sable')
    plt.title('Évolution d une image sur différents horizons')
if l < max :
    plt.figure()
    for i in range(0,l):
        plt.plot(prediction[l-i-1,i,:])
        string.append('horizon ' + str(i+1))
    plt.plot(yval1[l-1])
    string.append('truth')
    plt.legend(string)
    plt.xlabel('pixels')
    plt.ylabel('quantité de sable')
    plt.title('Évolution d une image sur différents horizons')   
    
# series temporelles des différents horizons (colonne de la matrice)
horizon=6
if visualisation:
    string=[]
    plt.figure()
    for i in range(0,horizon):
        plt.plot(prediction[:,i,25])
        string.append('horizon ' + str(i+1))
    plt.plot(yval1[:,25])
    string.append('truth')
    plt.legend(string)
    plt.xlabel('temps')
    plt.ylabel('concentration')
    plt.title('Séries chonologiques (pixel central)')
    
    
plt.show()
        