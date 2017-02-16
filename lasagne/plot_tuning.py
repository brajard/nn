#!/usr/bin/env python
import pickle
import matplotlib.pyplot as plt
from sklearn.base import clone
from datatools import nweights
from deepnn import kerasnn
import numpy as np
import os
   
outfile= 'output_gridsearch_allparam'

result = pickle.load(open(outfile+'.p','rb'))


nn0 = kerasnn(shapef_=(6,3,7,7))

params = result['params']
net_type = {p['network_type_'] for p in params}

nw = dict()
corr = dict()
corrapp = dict()


dlab = {'dense':'Dense','conv':'CNN','lstm':'LSTM','all':'CNN+LSTM'}
dcol = {'dense':'red','conv':'blue','lstm':'green','all':'black'}
lnet = ['dense','lstm','conv','all']
for ntype in net_type:
    lparam = [(p,m,app) for p,m,app in zip(params,result['mean_test_score'],result['mean_train_score']) \
                  if p['network_type_']==ntype]
    nw[ntype] = np.zeros(len(lparam))
    corr[ntype] = np.zeros(len(lparam))
    corrapp[ntype] = np.zeros(len(lparam))
   
    for i,(p,m,app) in enumerate(lparam):
        nn = clone(nn0)
        nn.set_params(**p)
        nn.set_shape()
        nw[ntype][i]=nweights(nn.set_model())
        corr[ntype][i]=m
        corrapp[ntype][i]=app
    plt.plot(nw[ntype],corr[ntype],'.',markersize=10,label=dlab[ntype],color=dcol[ntype])
    #print of the best
    ibest= np.argmax(corr[ntype])
    print(lparam[ibest],nw[ntype][ibest])
#plt.plot(nw,corrapp,'.',markersize=10,color='0.75',label='training')
#plt.plot(nw,corr,'.',markersize=10,color='black',label='validation')
plt.xlabel('Total number of weights')
plt.ylabel('correlation')
plt.legend(loc=0)
plt.savefig(outfile+'.png')

plotams = True
if plotams:
#Plot pour la AMS
    plt.close("all")
    zorder =10
    outdir = 'figams'
    os.makedirs(outdir,exist_ok=True)
    lnet = ['dense','lstm','conv','all']
    for ntype in lnet:
        plt.plot(nw[ntype],corr[ntype],'.',markersize=10,label=dlab[ntype],color=dcol[ntype],zorder=zorder)
        zorder = zorder -1

        plt.xlabel('Total number of weights')
        plt.ylabel('correlation')
        plt.xlim(0,8000)
        plt.ylim(0.8,0.96)
 #       plt.legend(loc=4)
        plt.savefig(os.path.join(outdir,outfile+ '_' + ntype +'.png'))
        
    lnet = ['conv','all']
    plt.close("all")
    for ntype in lnet:
        plt.plot(nw[ntype],corr[ntype],'.',markersize=10,label=dlab[ntype],color=dcol[ntype],zorder=zorder)
        zorder = zorder -1
    
    plt.xlabel('Total number of weights')
    plt.ylabel('correlation')
    plt.xlim(0,6000)
    plt.ylim(0.8,1.0)
#    plt.legend(loc=0)
    plt.savefig(os.path.join(outdir,outfile+ '_zoom.png'))



#masklim = np.logical_and(nw >2000,nw<2500)
#i = np.argmax(corr[masklim])
#iopt = int(np.argwhere(masklim)[i])
#paropt = result['params'][iopt]

