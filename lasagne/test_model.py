import os
import datatools
reload(datatools)
from datatools import plot_sequence,plot_scatter
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

outdir = '../data/nn_rec2'
modelname = 'rnn.json'
weights = 'weights.h5'
data =  'data.npz'

plt.close("all")

model = model_from_json(open(os.path.join(outdir,modelname)).read())
model.load_weights(os.path.join(outdir,weights))
npzfile = np.load(os.path.join(outdir,data))
Xapp = npzfile['Xapp']
yapp = npzfile['yapp']

start = -24*30


ypred = model.predict(Xapp).squeeze()

plot_sequence(yapp.reshape([-1,7,7]),ypred.reshape([-1,7,7]),start=start,length=7)

ypers = Xapp.squeeze()[:,-1,:].reshape(Xapp.shape[0],-1)
plot_sequence(yapp.reshape([-1,7,7]),ypers.reshape([-1,7,7]),start=start,length=7)

num_pix = 24
yapp_c = yapp[start:,num_pix]
ypred_c = ypred[start:,num_pix]
ypers_c = ypers[start:,num_pix]

print "NN performance:"
plot_scatter(yapp_c,ypred_c)

print "persistence performance:"
plot_scatter(yapp_c,ypers_c)
