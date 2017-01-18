import pickle
import matplotlib.pyplot as plt
from datatools import nweights
from deepnn import kerasnn
import numpy as np
   
outfile= 'output_gridserch_allparam'

result = pickle.load(open(outfile+'.p','rb'))


nn = kerasnn(shapef_=(6,3,7,7))

params = result['params']
net_type = {p['network_type_'] for p in params}

nw = dict()
corr = dict()
corrapp = dict()
for ntype in net_type:
    lparam = [(p,m,app) for p,m,app in zip(params,result['mean_test_score'],result['mean_train_score']) \
                  if p['network_type_']==ntype]
    nw[ntype] = np.zeros(len(lparam))
    corr[ntype] = np.zeros(len(lparam))
    corrapp[ntype] = np.zeros(len(lparam))
    for i,(p,m,app) in enumerate(lparam):
        nn.set_params(**p)
        nn.set_shape()
        nw[ntype][i]=nweights(nn.set_model())
        corr[ntype][i]=m
        corrapp[ntype][i]=app
    plt.plot(nw[ntype],corr[ntype],'.',markersize=10,label=ntype)


#plt.plot(nw,corrapp,'.',markersize=10,color='0.75',label='training')
#plt.plot(nw,corr,'.',markersize=10,color='black',label='validation')
plt.xlabel('Total number of weights')
plt.ylabel('correlation')
plt.legend(loc=0)
plt.savefig(outfile+'.png')

#masklim = np.logical_and(nw >2000,nw<2500)
#i = np.argmax(corr[masklim])
#iopt = int(np.argwhere(masklim)[i])
#paropt = result['params'][iopt]

