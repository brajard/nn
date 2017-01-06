#!/usr/bin/env python
from __future__ import print_function
import os
from importlib import reload
import datatools
reload(datatools)
from datatools import plot_sequence,plot_scatter
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import pickle
from keras.utils.visualize_util import plot

outdir = '../data/nn_rec3'
modelname = 'rnn.json'
weights = 'weights.h5'
data =  'dataval.npz'
histname = 'history.p'
tosave = True
tosavemat = True
tosavemodel = True

plt.close("all")

model = model_from_json(open(os.path.join(outdir,modelname)).read())
model.load_weights(os.path.join(outdir,weights))
if tosavemodel:
    plot(model, show_shapes = True, to_file=os.path.join(outdir,'model.png'))

npzfile = np.load(os.path.join(outdir,data))
Xapp = npzfile['Xval']
yapp = npzfile['yval']

start = -24*30


ypred = model.predict(Xapp).squeeze()

plot_sequence(yapp.reshape([-1,7,7]),ypred.reshape([-1,7,7]),start=start,length=7)

ypers = Xapp[:,-1,0].reshape(Xapp.shape[0],-1)
plot_sequence(yapp.reshape([-1,7,7]),ypers.reshape([-1,7,7]),start=start,length=7)

num_pix = 24
yapp_c = yapp[start:,num_pix]
ypred_c = ypred[start:,num_pix]
ypers_c = ypers[start:,num_pix]

print("NN performance:")
plot_scatter(yapp_c,ypred_c)

print("persistence performance:")
plot_scatter(yapp_c,ypers_c)

figtitle=['Seq_nn','Seq_pers','scatter_nn','scatter_pers']

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
