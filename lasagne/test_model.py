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

outdir = '../data/nn_rec4'
modelname = 'rnn.json'
weights = 'weights.h5'
data =  'data.npz'
tosave = True
tosavemat = True

plt.close("all")

model = model_from_json(open(os.path.join(outdir,modelname)).read())
model.load_weights(os.path.join(outdir,weights))
npzfile = np.load(os.path.join(outdir,data))
Xapp = npzfile['Xapp']
yapp = npzfile['yapp']

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

dmat={'Xapp':Xapp,'yapp':yapp,'ypred':ypred}
if tosavemat:
    savemat(os.path.join(outdir,'matfile.mat'),dmat)
