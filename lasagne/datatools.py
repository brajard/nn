from scipy.io import loadmat
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def load_dataset(
    datadir = '../data',
    fname = 'MATRICE_MAREE_11_2015.mat', 
    geofile = 'USABLE_PointbyPoint.mat',
    field = 'MATRICE', 
    geofield = 'USABLE_PointbyPoint1',
    linnum = [134,135]
    ):
    
    data = loadmat(os.path.join(datadir,fname))
    data = np.transpose(data[field][linnum,:])
    geo  = loadmat(os.path.join(datadir,geofile))
    geo = geo[geofield]

    Nt = len(geo[0,0].ravel())
    Np = len(linnum)
    X = np.zeros((Nt,Np)+geo.shape)
    
    for (i,j),ind in np.ndenumerate(geo):
        X[:,:,i,j] = data[ind.ravel(),:]
    return X

def plot_images(X,ind,titles=['Ubar','Vbar']):
    for j in range(len(ind)):
        fig,ax = plt.subplots(X.shape[1])
        for i in range(len(ax)):
            ax[i].imshow(X[ind[j],i])
            ax[i].set_title(titles[i]+'(' + str(ind[j]) +')')

def split_base(X,frac_train = 0.8,frac_test = 0.1, r_state = None,oprint = True):
    if r_state is None :
        r_state = random.random()

    N = X.shape[0]
    Lind = range(N)
    random.shuffle(Lind,lambda : r_state)
    Ntrain = int(frac_train*N)
    Ntest = int(frac_test*N)
    if oprint :
        print 'Learning set size : ',Ntrain,'/',N
        print 'Test set size : ',Ntest,'/',N
    print Lind[0:10]
    return X[Lind[:Ntrain]],X[Lind[-Ntest:]],Lind[:Ntrain],Lind[-Ntest:]
