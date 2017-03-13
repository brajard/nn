#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:09:48 2017

@author: cvasseur
"""
from __future__ import print_function
from scipy.io import loadmat
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from subprocess import call
import pandas as pd
import xarray as xr
from sklearn import preprocessing
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.pooling import MaxPooling2D
#from keras.layers.extra import TimeDistributedFlatten,TimeDistributedConvolution2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Deconvolution2D,Convolution2D
import theano
import pickle
from keras.optimizers import rmsprop
from datatools import prepare_data, make_train

plt.close("all")
outdir = '../data/nn_bestnet'

datadir = '/net/argos/data/parvati/aaclod/home2/aaclod/MAREES/python_nn/nn'
fname = 'MATRICE_01_2017.mat'
geofile = 'USABLE_PointbyPoint_01_2017.mat'
data,scaler = prepare_data(datadir=datadir,fname=fname,geofile=geofile,lognorm=True,epsi=0.0001,nseq=6,smNum = [44,55,66,77,88,99,110], uvNum=[])


model = Sequential()
