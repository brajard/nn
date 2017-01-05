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

colindex = [str(i) for i in range(136)]
colindex[134]='ubar'
colindex[135]='vbar'

def normalize (X,scaler=preprocessing.StandardScaler,fit=True,parname = 'parameters'):
    z = [c for c in X.coords.dims if c != parname]
    stacked = X.stack(z=z).T
    if fit:
        scaler.fit(stacked)
    stacked.values = scaler.transform(stacked.values)
    L = X.coords.dims
    Xn = stacked.unstack('z')
    if len(L)==4:
         Xn = Xn.transpose(L[0],L[1],L[2],L[3])
    if len(L)==5:
         Xn = Xn.transpose(L[0],L[1],L[2],L[3],L[4])
    return(Xn,scaler)

def denormalize (X,scaler=preprocessing.StandardScaler,parname = 'parameters'):
    z = [c for c in X.coords.dims if c != parname]
    stacked = X.stack(z=z).T
    stacked.values = scaler.inverse_transform(stacked.values)
    L = X.coords.dims
    Xd = stacked.unstack('z')
    if len(L)==4:
         Xd = Xd.transpose(L[0],L[1],L[2],L[3])
    if len(L)==5:
         XD = Xd.transpose(L[0],L[1],L[2],L[3],L[4])
    return(Xd,scaler)

    
def load_sequence(
    data,
    n_prev = 3,
    hor=1):
    X,Y = [],[]
    for i in range(len(data)-n_prev-hor+1):
        X.append(data[i:i+n_prev])
        Y.append(data[i+n_prev+hor-1])
    return np.array(X),np.array(Y)

def load_dataset(
    datadir = '../data',
    fname = 'MATRICE_MAREE_06_2016.mat', 
    geofile = 'USABLE_PointbyPoint.mat',
    field = 'MATRICE', 
    geofield = 'USABLE_PointbyPoint1',
    linnum = [134,135], #ubar, vbar
    start = '20070101',
    cols = colindex,
    outputformat = 'np'
    ):
    
    if not os.path.isfile(os.path.abspath(os.path.join(datadir,fname))):
        url = 'http://skyros.locean-ipsl.upmc.fr/~jbrlod/download/MATRICE_MAREE_06_2016.mat'
#        url = 'https://www.dropbox.com/sh/9rg6nrn8xhk5wjz/AACOB1QGXAFT4w9dnsujefsTa/MATRICE_MAREE_11_2015.mat'
        #call(["wget", "-P "+os.path.abspath(datadir), url],shell=True)
        os.system("wget " + "-P " + os.path.abspath(datadir) + " " + url)
    
    cols = np.array(cols)
    data = loadmat(os.path.join(datadir,fname))
    data = np.transpose(data[field][linnum,:])
    geo  = loadmat(os.path.join(datadir,geofile))
    geo = geo[geofield]


    
    Nt = len(geo[0,0].ravel())
    Np = len(linnum)

    dates = pd.date_range(start,dtype='datetime64[ns]',periods=Nt,freq='H')
    xind,yind = range(geo.shape[0]), range(geo.shape[1]) #TODO ask anastase which is x, which is y

    X = np.zeros((Nt,Np)+geo.shape)

    for (i,j),ind in np.ndenumerate(geo):
        X[:,:,i,j] = data[ind.ravel()-1,:]

    dX = xr.DataArray(X,coords=[dates,cols[linnum],xind,yind],dims=['dates','parameters','xind','yind'])
    if outputformat == 'np':
        return X
    else:
        return dX

def plot_images(X,ind,titles=['Ubar','Vbar']):
    for j in range(len(ind)):
        fig,ax = plt.subplots(X.shape[1])
        for i in range(len(ax)):
            pc = ax[i].imshow(X[ind[j],i])
            ax[i].set_title(titles[i]+'(' + str(ind[j]) +')')
            plt.colorbar(pc)

def split_base(X,frac_train = 0.8,frac_test = 0.1, r_state = None,oprint = True):
    if r_state is None :
        r_state = random.random()

    N = X.shape[0]
    Lind = range(N)
    random.shuffle(Lind,lambda : r_state)
    Ntrain = int(frac_train*N)
    Ntest = int(frac_test*N)
    if oprint :
        print('Learning set size : ',Ntrain,'/',N)
        print('Test set size : ',Ntest,'/',N)
    print (Lind[0:10])
    return X[Lind[:Ntrain]],X[Lind[-Ntest:]],Lind[:Ntrain],Lind[-Ntest:]

def plot_compare1(X,X_predict,ind):
    for j in range(len(ind)):
        plt.figure()
        plt.subplot(311)
        plt.imshow(X[ind[j],0],interpolation='nearest')
        plt.title('Ubar true')
        plt.colorbar()
        plt.subplot(312)
        plt.imshow(X[ind[j],1],interpolation='nearest')
        plt.title('Vbar true')
        plt.colorbar()
        plt.subplot(313)
        plt.imshow(X_predict[ind[j],0],interpolation='nearest')
        plt.title('Ubar predict')
        plt.colorbar()
        plt.subplot(324)
        plt.imshow(X_predict[ind[j],1],interpolation='nearest')
        plt.title('Vbar predict')
        plt.colorbar()
        plt.subplot(325)
        plt.imshow(X_predict[ind[j],0]-X[ind[j],0],interpolation='nearest')
        plt.title('err Ubar')
        plt.colorbar()
        plt.subplot(326)
        plt.imshow(X_predict[ind[j],1]-X[ind[j],1],interpolation='nearest')
        plt.title('err Vbar')
        plt.colorbar()
        

def plot_compare(X,X_predict,ind):
    for j in range(len(ind)):
        plt.figure()
        plt.subplot(311)
        plt.imshow(X[ind[j]],interpolation='nearest')
        plt.title('true')
        plt.colorbar()
        plt.subplot(312)
        plt.imshow(X_predict[ind[j]],interpolation='nearest')
        plt.title('predict')
        plt.colorbar()
        plt.subplot(313)
        plt.imshow(X_predict[ind[j]]-X[ind[j]],interpolation='nearest')
        plt.title('diff')
        plt.colorbar()


def plot_sequence(yt,yr,start=0,length=5):
    f,ax = plt.subplots(3,length)
    for it in range(length):
        ax[0,it].imshow(yt[start+it], interpolation='nearest')
#        plt.colorbar(ax[0,it])
        ax[1,it].imshow(yr[start+it], interpolation='nearest')
#        plt.colorbar()
        ax[2,it].imshow(yr[start+it]-yt[start+it], interpolation='nearest')
#        plt.colorbar()
    
def plot_scatter(yt,yr):
    plt.figure()
    plt.scatter(yt,yr)
    corr = np.corrcoef(yt,yr)
    print ("correlation=",corr[0,1])
    rmse = np.sqrt(np.mean((yt-yr)**2))
    print ("rmse=",rmse)

def history2dict(history):
    """Transform the history callback return by the 
    fit method to a dictionnary with fields :
    epoch, history, params"""
    return {'history':history.history,'params':history.params,'epoch':history.epoch}
    

def prepare_data(smNum=[77],uvNum=[134,135],nseq = 6, epsi = 1e-3, first_time = 8*30*24, lognorm = True,itest=[]):
    """ prepare data for learning, and test function
    
    Parameters:
    -----------
    smNum : list[int]
            index of columns in dataset for suspended matter values (to be sum in the learning dataset
    uvNum : list[int]
            index of columns in dataset for other parameters
    nseq : int
           size of the sequence in input
    epsi : float
           minimum value for sm parameter
    first_time : int
                 index of the first time step to considere in the learning
    lognorm : bool
              True if sm parameter should be log10 normalized
    itest : list[int]
            index of rows in dataset for test
    
    Returns :
    ---------
    Xapp,yapp : tuple(np.array,np.array)
                learning dataset
    Xval,yval : tuple(np.array,np.array)
                  test dataset (can be empty)
    scaler : scaler from sklearn
    """ 
    linnum = smNum + uvNum
    Xtmp = load_dataset(linnum=linnum,outputformat='xr')
    nt = Xtmp['dates'].size
    X2   = Xtmp.isel(dates=slice(first_time,None))
    Xtsm = X2.isel(parameters=slice(len(smNum)))
  #  Xtsm = np.sum(Xtmp[first_time:,:len(smNum),:,:],axis=1,keepdims=True)
    Xtsm = Xtsm.sum(dim = 'parameters')
    Xtsm.values[Xtsm.values<epsi]=epsi
    if lognorm:
        Xtsm.values = np.log10(Xtsm.values)

    Xuv = X2.isel(parameters=slice(len(smNum),None))
    
    #definition new DataArray
    coords = {c:Xuv.coords.get(c).values for c in Xuv.coords.dims}
    coords['parameters']=np.insert(coords['parameters'],0,'tsm')
    nt,npar,nx,ny = Xuv.shape
    X = np.concatenate((Xtsm.values.reshape((nt,1,nx,ny)),Xuv.values),axis=1)
    X = xr.DataArray(X,coords=coords,dims=Xuv.coords.dims)

#    X = np.concatenate((Xtsm,Xuv),axis=1)
    nt,npar,nx,ny = X.shape
    scaler = preprocessing.StandardScaler()
    Xn,scaler = normalize(X,scaler)
    
    XX,y = load_sequence(Xn.values,n_prev=nseq)
    y = y[:,0,:,:]
    n = XX.shape[0]
    ival = list(range(0,21*24)) + list(range(n-21*24,n))
    ilearn = [i for i in range(n) if i not in ival]
    
    XX = XX.reshape([XX.shape[0],nseq,npar,nx,ny])
    y = y.reshape([y.shape[0],nx*ny])

    Xapp = XX[ilearn]
    yapp = y[ilearn]
    Xval = XX[ival]
    yval = y[ival]


    return Xapp,yapp,Xval,yval,scaler
    
