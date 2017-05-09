#!/usr/bin/env python
from __future__ import print_function
import os
from importlib import reload
import datatools
reload(datatools)
from datatools import plot_sequence,plot_scatter, load_nc, save_nc
from keras.models import model_from_json
import matplotlib.pyplot as plt
from scipy.io import savemat
import pickle
import xarray as xr
from keras.utils.visualize_util import plot

outdir = '../data/nn_sst'
modelname = 'rnn.json'
weights = 'weights.h5'
data =  'dataval.npz'
histname = 'history.p'
fdata = '../data/data_sst_keras'
tosave = True
tosavemat = True
tosavemodel = True
plt.close("all")

model = model_from_json(open(os.path.join(outdir,modelname)).read())
model.load_weights(os.path.join(outdir,weights))
if tosavemodel:
    plot(model, show_shapes = True, to_file=os.path.join(outdir,'model.png'))

data = load_nc(fdata)
Xapp = data['Xval']
yapp = data['yval'].squeeze()



start = 0
iparam = 0

#dims shape :
nt,nseq,npar,nx,ny = Xapp.shape
#name of dims :
date_n,seq_n,par_n,xind_n,yind_n = Xapp.dims
date_n,pixind_n = yapp.dims

#Compute persistence
ypers = Xapp[:,-1,0].stack(**{pixind_n:(xind_n,yind_n)})
ypers.name = 'ypers'
seq_true = yapp.unstack(date_n).unstack(pixind_n)
seq_pers = ypers.unstack(date_n).unstack(pixind_n)
plot_sequence(seq_true[:,iparam,:,:].values,seq_pers[:,iparam,:,:].values)




if 'conv' in outdir or 'dense' in outdir:
    #TODO test it
    #Xapp = Xapp.reshape(tuple(nt)+ tuple([nseq*npar])+(nx,ny))
    Xapp = Xapp.stack(sample = (date_n,seq_n,par_n))
ypred_np = model.predict(Xapp).squeeze()

ypred = ypers.copy()
ypred.name = 'ypred'
ypred.values = ypred_np
seq_pred = ypred.unstack(date_n).unstack(pixind_n)

plot_sequence(seq_true[:,iparam,:,:].values,seq_pred[:,iparam,:,:].values)

num_pix = nx*ny//2 + ny
yapp_c = yapp[start:,num_pix]
ypred_c = ypred[start:,num_pix]
ypers_c = ypers[start:,num_pix]

print("NN performance:")
plot_scatter(yapp_c,ypred_c)

print("persistence performance:")
plot_scatter(yapp_c,ypers_c)

figtitle=['Seq_pers','Seq_nn','scatter_nn','scatter_pers']

for i in range(1,5):
    plt.figure(i)
    plt.title(figtitle[i-1])
    if tosave:
        plt.savefig(os.path.join(outdir,figtitle[i-1]+'.png'))

plt.figure()
hist = pickle.load(open(os.path.join(outdir,histname),'rb'))
plt.plot(hist['epoch'],hist['history']['loss'],color='0.75',linewidth=2.0,label='Learning loss')
plt.plot(hist['epoch'],hist['history']['val_loss'],color='black',linewidth=2.0,label='validation loss')
plt.title('Learning')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend()
if tosave:
    plt.savefig(os.path.join(outdir,'learning.png'))

dmat={'Xapp':Xapp,'yapp':yapp,'ypred':ypred}
if tosavemat:
    savemat(os.path.join(outdir,'matfile.mat'),dmat)
    #TO DO : rename xind, yind dimensions
    ds = xr.merge([seq_pers,seq_pred,seq_true])
    save_nc(ds,os.path.join(outdir,'outval'))