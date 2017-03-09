#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:47:39 2017

@author: cvasseur
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

def plot_scatterbis(yt,yr):
    #plt.figure()
    #plt.scatter(yt,yr)
    corr = np.corrcoef(yt,yr)
    print ("correlation=",corr[0,1])
    rmse = np.sqrt(np.mean((yt-yr)**2))
    print ("rmse=",rmse)
    return corr[0,1],rmse

def moy_corr(yv,ypred,horizon):
    corr=0
    n=0
    for i in range(0,len(ypred)-horizon+1):
        corr=corr+np.corrcoef(ypred[i],yv[i+horizon-1])[1,0]
        n=n+1
    return corr/n


def plot_horizon(corr):
    plt.figure()
    plt.plot(corr)
    plt.title('Corrélation en fonction de l horizon')
    plt.xlabel('horizon')
    plt.ylabel('corrélation')
    #plt.show()
    
