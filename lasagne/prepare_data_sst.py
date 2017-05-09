#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:30:24 2017

@author: jbrlod
"""
#%% Init
import xarray as xr
import os
import numpy as np
from datatools import load_sequence,normalize, save_nc
from sklearn import preprocessing
import pickle

#%% prepare_data_sst
nseq = 6
first_time = 2*30
datadir = '../data'
fname = 'sst_norm.nc'
outname = 'data_sst_keras'
Xtmp = xr.open_dataset(os.path.join(datadir,fname))
data_names = [k for k in Xtmp.data_vars]
L = [Xtmp[fname] for fname in data_names]
Xtmp = xr.concat(L,dim='parameters')
Xtmp['parameters']=('parameters',data_names)
Xtmp=Xtmp.expand_dims('par') #Simension 1 chanel
Xtmp['par']=('par',[0])

Xtmp = Xtmp.transpose('dates','parameters','par','xind','yind')
Xtmp.name = 'thetao'
#%% Process normalizartion

scaler = preprocessing.StandardScaler()
Xn,scaler = normalize(Xtmp,scaler=scaler,parname='par')


#%% Extract
XX,y = load_sequence(Xn,n_prev=nseq)

n = XX.shape[0] #number of samples (dates)
    
   
ival = list(range(0,first_time)) + list(range(n-first_time,n))
ilearn = [i for i in range(n) if i not in ival]

#   XX = XX.reshape([XX.shape[0],nseq,npar,nx,ny])
y = y.rename({'xind':'xind_y','yind':'yind_y'}) 
#to avoid merge conflict
y = y.stack(pixind=('xind_y','yind_y'))

Xapp = XX[ilearn]
yapp = y[ilearn]
Xval = XX[ival]
yval = y[ival]



Xapp = Xapp.stack(dates_app=('dates','parameters'))
Xapp = Xapp.transpose('dates_app','seq','par','xind','yind')

Xval = Xval.rename({'dates':'datesV','parameters':'parametersV'})
Xval = Xval.stack(dates_val=('datesV','parametersV'))
Xval = Xval.transpose('dates_val','seq','par','xind','yind')

yapp = yapp.stack(dates_app=('dates','parameters'))
yapp = yapp.transpose('dates_app','par','pixind')

yval = yval.rename({'dates':'datesV','parameters':'parametersV'})
yval = yval.stack(dates_val=('datesV','parametersV'))
yval = yval.transpose('dates_val','par','pixind')

#Creating the dataset 
Xapp.name = 'Xapp'
yapp.name = 'yapp'
Xval.name = 'Xval'
yval.name = 'yval'
data = xr.merge([Xapp,yapp,Xval,yval])



#%% Save
save_nc(data,os.path.join(datadir,outname))

