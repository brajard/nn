#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:47:39 2017

@author: cvasseur
"""

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

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

# prediction des temps t+1,t+2,t+3 ... t+max
# sauvegarde des resultats dans le dossier data/prediction/
def predict_time(model,Xval,yval,prediction,max,look_back,outdir,fname):
    size=len(prediction)
    for j in range(0,max-1):
        prediction[:,j,:] = model.predict(Xval).squeeze()
    
        # Maj de Xval
        for i in range(1,look_back-1):
            Xval[:,i-1]=Xval[:,i]
        Xval[:,look_back-1]=prediction[:,j,:].reshape([size,1,7,7])
        
        prediction[:,j+1,:] = model.predict(Xval).squeeze()
    
    # moyenne des corrélations par colonne de la matrice prediction avec les valeurs de yval(ici yapp)
    corr = np.zeros(max)
    for i in range(0,max):
        corr[i] = moy_corr(yval,prediction[:,i],i+1)
    
    # sauvegarde des fichiers
    prediction=xr.DataArray(prediction)
    prediction.name = 'prediction'
    corr=xr.DataArray(corr)
    corr.name='correlation'
    data=xr.merge([prediction, corr])
    
    os.makedirs(outdir,exist_ok=True)
    np.savez(os.path.join(outdir,fname),prediction=data.prediction,corr=data.correlation)

def calcul_persistence(Ypers,Yval,max):
    corr_hor=np.zeros(max)
    for j in range(0,max):
        correlation=0
        cmpt=0
        for i in range(0,len(Ypers)-j):
            correlation=np.corrcoef(Yval[i+j,],Ypers[i,])[0,1]+correlation
            cmpt=cmpt+1
        corr_hor[j]=correlation/cmpt    
    return corr_hor

def plot_horizon(corr,corr2,title):
    plt.figure()
    plt.plot(corr)
    plt.plot(corr2)
    plt.title(title)
    plt.legend(['modele','persistence'])
    plt.xlabel('horizon')
    plt.ylabel('corrélation')
    #plt.show()

def plot_ligne_prediction(prediction,yval1,l,max,title):
    string = []
    if l > max :
        plt.figure()
        plt.plot(yval1[l-1])
        string.append('truth')
        for i in range(0,max):
            plt.plot(prediction[l-i-1,i,:])
            string.append('horizon ' + str(i+1))
        plt.legend(string)
        plt.xlabel('pixels')
        plt.ylabel('quantité de sable')
        plt.title(title)
    if l < max :
        plt.figure()
        plt.plot(yval1[l-1])
        string.append('truth')        
        for i in range(0,l):
            plt.plot(prediction[l-i-1,i,:])
            string.append('horizon ' + str(i+1))
        plt.legend(string)
        plt.xlabel('pixels')
        plt.ylabel('quantité de sable')
        plt.title(title)   


def plot_temporelle(prediction,yval1,horizon,title):
    string=[]
    size=len(prediction)
    plt.figure()
    plt.plot(yval1[:,25])
    string.append('truth')    
    for i in range(0,horizon):
        a=np.zeros(size)
        a[i:size]=prediction[0:size-i,i,25]
        plt.plot(a)
        string.append('horizon ' + str(i+1))
    plt.legend(string)
    plt.xlabel('temps')
    plt.ylabel('concentration')
    plt.title(title)