import pickle
import matplotlib.pyplot as plt
from datatools import define_model_all,nweights
import numpy as np
   
outfile= 'output_gridsearch_step3'

corrpar = {'shape_':'shape','n_feat_':'n_feat','filter_size_':'filter_size','nhid1_':'nhid1',
           'nhid2_':'nhid2','pool_size_':'pool_size','lr_':'lr'}

result = pickle.load(open(outfile+'.p','rb'))

shape = (1,6,3,7,7)

nw = np.zeros(len(result['params']))
corr = np.zeros(len(result['params']))
corrapp = np.zeros(len(result['params']))

for i,(p,m,mapp) in enumerate(zip(result['params'],result['mean_test_score'],result['mean_train_score'])):
    ipar = {corrpar[k]:v for k,v in p.items() if k in corrpar}
    nn = define_model_all(shape,**ipar)
    nw[i] = nweights(nn)
    corr[i] = m
    corrapp[i] = mapp

for n,m,mmapp in zip(nw,corr,corrapp):
    plt.plot([n,n],[m,mmapp],':',color='0.5')



plt.plot(nw,corrapp,'.',markersize=10,color='0.75',label='training')
plt.plot(nw,corr,'.',markersize=10,color='black',label='validation')
plt.xlabel('Total number of weights')
plt.ylabel('correlation')
plt.legend(loc=0)
plt.savefig(outfile+'.png')

masklim = np.logical_and(nw >2000,nw<2500)
i = np.argmax(corr[masklim])
iopt = int(np.argwhere(masklim)[i])
paropt = result['params'][iopt]

