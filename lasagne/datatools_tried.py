#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:47:39 2017

@author: cvasseur
"""
from matplotlib import gridspec
#from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def create_new_data(data,data2):
    Xapp = np.concatenate((data.Xapp,data2.Xapp),axis=0)
    yapp = np.concatenate((data.yapp,data2.yapp),axis=0)
    #dates = np.concatenate((data.Xapp.dates,data2.Xapp.dates),axis=0)
    
    Xapp = xr.DataArray(Xapp)
    Xapp.name = 'Xapp'
    yapp = xr.DataArray(yapp)
    yapp.name = 'yapp'
    Xval = xr.DataArray(data.Xval)
    Xval.name = 'Xval'
    yval = xr.DataArray(data.yval)
    yval.name = 'yval'
    
    res=xr.merge([Xapp,yapp,Xval,yval])
    return res
    
def new_data(model, data):
    size, lookback, npar, nx ,ny = data.Xapp.shape
    
    prediction = model.predict(data.Xapp).squeeze()
    
    data.Xapp[:,:-1]=data.Xapp[:,1:]
    data.Xapp[:,-1]=prediction[:,:].reshape([size,1,nx,ny])    
    
    Xapp = xr.DataArray(data.Xapp[:size-1,:])
    Xapp.name = 'Xapp'
    yapp = xr.DataArray(data.yapp[1:,:])
    yapp.name = 'yapp'
    Xval = xr.DataArray(data.Xval)
    Xval.name = 'Xval'
    yval = xr.DataArray(data.yval)
    yval.name = 'yval'
    
    Xapp['dates']=yapp['dates']
    
    data = xr.merge([Xapp, yapp, Xval, yval])
    return data

def move_data(Xval, prediction,i):
     size, lookback, npar, nx ,ny = Xval.shape
        
     Xval[:,:-1]=Xval[:,1:]
     Xval[:,-1]=prediction[:,i,:].reshape([size,npar,nx,ny])   
     
     return Xval
    
def plot_scatterbis(yt,yr):
    #plt.figure()
    #plt.scatter(yt,yr)
    corr = np.corrcoef(yt,yr)
    print ("correlation=",corr[0,1])
    rmse = np.sqrt(np.mean((yt-yr)**2))
    print ("rmse=",rmse)
    return corr[0,1],rmse

def moy_rmse(yv, ypred, horizon):
    rmse=0
    n=0
    for i in range(0,len(ypred)-horizon+1):
        rmse=rmse+np.sqrt(np.mean((yv[i+horizon-1]-ypred[i])**2))
        n=n+1
    return rmse/n

def moy_corr(yv,ypred,horizon):
    corr=0
    n=0
    for i in range(0,len(ypred)-horizon+1):
        corr=corr+np.corrcoef(ypred[i],yv[i+horizon-1])[1,0]
        n=n+1
    return corr/n

def jointPlot(ypred,fname,title):
    x = np.zeros(7)
    y = np.zeros(7)
    bins=np.arange(7)
    
    for i in range(0,7):
        x[i] = np.mean(ypred[i,:])
        y[i] = np.mean(ypred[:,i])    
        
    #Define grid for subplots
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios = [1, 4])

    #Create img plot
    fig = plt.figure(facecolor='white')
    ax = plt.subplot(gs[1, 0],frameon = False)
    plt.imshow(ypred)
    plt.colorbar(orientation ='vertical')
    ax.grid(True)
    #Create Y-marginal (right)
    axr = plt.subplot(gs[1, 1], sharey=ax, frameon = False,xticks = [] ) #xlim=(0, 1), ylim = (ymin, ymax) xticks=[], yticks=[]
    axr.barh(bins,x, color = '#5673E0')

    #Create X-marginal (top)
    axt = plt.subplot(gs[0,0], sharex=ax,frameon = False,yticks = [])# xticks = [], , ) #xlim = (xmin, xmax), ylim=(0, 1)
    axt.bar(bins,y, color = '#5673E0')

    axt.set_title(title)
    ax.set_xlabel('pixels')
    ax.set_ylabel('pixels')

    #Bring the marginals closer to the scatter plot
    fig.tight_layout(pad = 1)
    plt.show()
    fig.savefig(fname)

# prediction des temps t+1,t+2,t+3 ... t+max
# + sauvegarde des resultats dans le dossier data/prediction/
def predict_time(model,Xval,yval,max,outdir,fname):
    size,lookback,npar,nx,ny = Xval.shape

    prediction = np.zeros((size,max,yval.shape[1]))
    for j in range(0,max):
        prediction[:,j,:] = model.predict(Xval).squeeze()
    
        # Maj de Xval
        Xval[:,:-1]=Xval[:,1:]
        Xval[:,-1]=prediction[:,j,:].reshape([size,npar,nx,ny]) 
    
    # moyenne des corrélations par colonne de la matrice prediction avec les valeurs de yval(ici yapp)
    corr = np.zeros(max)
    rmse = np.zeros(max)
    for i in range(0,max):
        corr[i] = moy_corr(yval,prediction[:,i],i+1)
        rmse[i] = moy_rmse(yval, prediction[:,i],i+1)
        #rmse[i] = np.sqrt(np.mean((yval-prediction[:,i])**2))
    
    # sauvegarde des fichiers
    prediction=xr.DataArray(prediction)
    prediction.name = 'prediction'
    corr=xr.DataArray(corr)
    corr.name='correlation'
    rmse=xr.DataArray(rmse)
    rmse.name = 'rmse'
    data=xr.merge([prediction, corr, rmse])
    
    os.makedirs(outdir,exist_ok=True)
    np.savez(os.path.join(outdir,fname),prediction=data.prediction,corr=data.correlation, rmse=data.rmse)
    
    return corr, rmse, prediction

# prediction pour un modele et calcul de corrélation, rmse
def predict_model(model,Xval,yval,index):
    size,lookback,npar,nx,ny = Xval.shape
    
    prediction = model.predict(Xval).squeeze()
    
    corr = moy_corr(yval,prediction,index+1)
    rmse = moy_rmse(yval, prediction,index+1)
    
    return prediction, corr, rmse
    
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

def plot_horizon(corr,corr2,rmse,rmse2,fname):
    
    fig, (ax1,ax2) = plt.subplots( nrows=1, ncols=2 )  # create figure & 1 axis
    ax1.plot(corr)
    ax1.plot(corr2)
    ax1.legend(['21 jours avant','21 jours après'])
    ax1.set_xlabel('horizon')
    ax1.set_ylabel('corrélation')
    ax1.set_title('Corrélation en fonction de l horizon')
    ax2.plot(rmse)
    ax2.plot(rmse2)
    ax2.legend(['21 jours avant','21 jours après'])
    ax2.set_xlabel('horizon')
    ax2.set_ylabel('rmse')    
    ax2.set_title('RMSE en fonction de l horizon')
    #plt.show()

    fig.savefig(fname)   # save the figure to file
    

def plot_ligne_prediction(prediction,yval1,l,title,fname):
    # l : ligne de la matrice predidction (defini également l'horizon)
    string = []
    max = prediction.shape[1]
    
    if l > max :
        fig = plt.figure()
        plt.plot(yval1[l-1])
        string.append('truth')
        for i in range(0,max):
            plt.plot(prediction[l-i-1,i,:])
            string.append('horizon ' + str(i+1))
        plt.legend(string)
        plt.xlabel('pixels')
        plt.ylabel('quantité de sable')
        plt.title(title)
    if l <= max :
        fig = plt.figure()
        plt.plot(yval1[l-1])
        string.append('truth')        
        for i in range(0,l):
            plt.plot(prediction[l-i-1,i,:])
            string.append('horizon ' + str(i+1))
        plt.legend(string)
        plt.xlabel('pixels')
        plt.ylabel('quantité de sable')
        plt.title(title)
        
    fig.savefig(fname)


def plot_temporelle(prediction,yval1,horizon,title,fname):
    string=[]
    size=len(prediction)
    
    fig = plt.figure()
    plt.plot(yval1[:,25])
    string.append('truth')    
    if 1:
        for i in range(0,horizon):
            a=np.zeros(size)
            a[i:size]=prediction[0:size-i,i,25]
            plt.plot(a)
            string.append('horizon ' + str(i+1))
    
    plt.legend(string)
    plt.xlabel('temps')
    plt.ylabel('concentration')
    plt.title(title)
    
    fig.savefig(fname)
