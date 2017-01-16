from deepnn import kerasnn
from sklearn.model_selection import GridSearchCV
from datatools import prepare_data
import pickle


outdir = '../data/nn_rec1'

datadir = '/net/argos/data/parvati/aaclod/home2/aaclod/MAREES/python_nn/nn'
fname = 'MATRICE_01_2017.mat'
geofile = 'USABLE_PointbyPoint_01_2017.mat'

data,scaler = prepare_data(datadir=datadir,fname=fname,geofile=geofile,lognorm=True,epsi=0.0001,smNum = [44,55,66,77,88,99,110])

X = data.Xapp.stack(z=data.Xval.dims[1:])
y = data.yapp
sk_nn = kerasnn(shapef_=data.Xval.shape[1:],nb_epoch_=100,validation_split_=0)
step=3
if step==1:
    outfile = 'output_gridsearch_step1'
    param_grid={'n_feat_':[3,5,7,10],'nhid1_':[8,12,16],'nhid2_':[4,6,8],'filter_size':[3,5]}

elif step==2:
    #TOP 3
#{'filter_size': 3, 'n_feat_': 10, 'nhid1_': 16, 'nhid2_': 8}
#{'filter_size': 5, 'n_feat_': 7, 'nhid1_': 12, 'nhid2_': 6}
#{'filter_size': 5, 'n_feat_': 10, 'nhid1_': 12, 'nhid2_': 6}
    
    outfile = 'output_gridsearch_step2'
    lr = [0.1,0.01,0.001,0.0001]
    batchsize = [128,256]
    param_grid=[{'filter_size': [3], 'n_feat_': [10], 'nhid1_': [16], 'nhid2_': [8], 'lr_':lr,'batchsize_':batchsize},
                {'filter_size': [5], 'n_feat_': [7], 'nhid1_': [12], 'nhid2_': [6], 'lr_':lr,'batchsize_':batchsize},
                {'filter_size': [5], 'n_feat_': [10], 'nhid1_': [12], 'nhid2_': [6], 'lr_':lr,'batchsize_':batchsize}]
elif step==3:
    #TOP 1
    # {'lr_': 0.01, 'n_feat_': 10, 'filter_size': 3, 'batchsize_': 256, 'nhid1_': 16, 'nhid2_': 8}

    outfile= 'output_gridsearch_step3'
    param_grid = {'n_feat_':[3,5,7,10],'nhid1_':[8,12,16,20],'nhid2_':[4,6,8,10],'filter_size_':[3],'lr_':[0.01],'batchsize_':[256]}

elif step==4:
    outfile= 'output_gridsearch_step4'
    param_grid = {'n_feat_':[3],'nhid1_':[8],'nhid2_':[4],'filter_size_':[5],'lr_':[0.01],'batchsize_':[256]}

clf = GridSearchCV(sk_nn,param_grid,cv=4,n_jobs=4,verbose=2)

clf.fit(X,y)
ostr =[]
ostr.append("Best parameters set found on development set:")
#ostr.append("")
ostr.append(str(clf.best_params_))
ostr.append("")
ostr.append("Grid scores on development set:")
#ostr.append("")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    ostr.append("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
#   ostr.append("")

ostr = '\n'.join(ostr)
print(ostr)
with open(outfile+'.txt','w') as f:
    f.write(ostr)
pickle.dump(clf.cv_results_,open(outfile+'.p','wb'))
