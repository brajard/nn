#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:13:43 2017

@author: cvasseur
"""

#!/usr/bin/env python
### Script : comparaison avec un jour moyen

## Import libraries
from datatools import prepare_data
import os
import numpy as np
import matplotlib.pyplot as plt


outdir = '../data/nn_bestnet'
data =  'dataval.npz'

plt.close("all")
outdir = '../data/nn_bestnet'

datadir = '/net/argos/data/parvati/aaclod/home2/aaclod/MAREES/python_nn/nn'
fname = 'MATRICE_01_2017.mat'
geofile = 'USABLE_PointbyPoint_01_2017.mat'
data,scaler = prepare_data(datadir=datadir,fname=fname,geofile=geofile,lognorm=True,epsi=0.0001,smNum = [44,55,66,77,88,99,110], uvNum=[])

# calcul du jour moyen (dupliqué en 5 jours)
Yapp=data.yapp[18:10746,:]
t=np.arange(0,10725,24)
res=np.zeros((120,49))
for i in range(0,24):
    res[i,:]=sum(Yapp[t+i])/447
    res[i+24,:]=res[i,:]
    res[i+48,:]=res[i,:]
    res[i+72,:]=res[i,:]
    res[i+96,:]=res[i,:]

plt.figure()
plt.plot(res)
plt.show()

#npzfile = np.load(os.path.join(outdir,data))
#Yval = npzfile['yval'][18:,:]
#Xval = npzfile['Xval']
#cmpt=0
#correlation=0
#for i in range(0,120):
#    correlation=np.corrcoef(Yval[i,],res[i,])[0,1]+correlation
#    cmpt=cmpt+1
#print(correlation/cmpt)