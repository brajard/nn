"""
This is a module to make a equivalent to scikit-learn estimator
with a deep neural network implemented in Keras"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from operator import mul
from functools import reduce
from datatools import define_model_all

#Some remarks
#all parameters should be named the same way in __init__method and for attributes
#they should all be keywords arguments

class kerasnn (BaseEstimator, RegressorMixin):
    """ Regressor interfaced the keras method to be usable in scikitlearn
    
    Parameters
    __________
    shape : shape of the keras input data (product of sape = n_features)
    kwargs : parameters of the model
    define_nn : keras model function (should return a keras model
    
    Attributes
    __________
    shapef_: shape of the keras input features (product of shapef = n_features)
    nn_ : keras model
    shape : shape of the keras input data (n_samples,shape_f)
    train_parameters_ : parameter for the training
    X_ : array, shape = [n_samples, n_features]
    The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
    The labels passed during :meth:`fit`"""

    def __init__(self,shapef_=None,n_feat_=5,filter_size_=3,nhid1_=12,
                 nhid2_=12,pool_size_=(2,2),lr_=0.001,
                 batch_size_=256,nb_epoch_=50,validation_split_=0.05,init_=0):
        if not shapef_:
            raise ValueError('shapef_ argument not set')
        self.shapef_ = shapef_
        self.n_feat_ = n_feat_
        self.filter_size_ = filter_size_
        self.nhid1_ = nhid1_
        self.nhid2_ = nhid2_
        self.pool_size_ = pool_size_
        self.batch_size_ = batch_size_
        self.nb_epoch_ = nb_epoch_
        self.validation_split_ = validation_split_
        self.init_=init_
        self.lr_=lr_
        self.paramset_ = {'shapef_','n_feat_','filter_size_','nhid1_','nhid2_','init_',
                          'pool_size_','batch_size_','nb_epoch_','validation_split_','lr_'}

    def reshape(self,X):
        nfeature = reduce(mul,self.shapef_)
        assert(nfeature==X.shape[1]),'wrong size of X'
        self.shape_ = tuple([X.shape[0]])+tuple(self.shapef_) 
        self.XX_ = X.reshape(self.shape_)

    def fit(self,X,y):
        X, y = check_X_y(X,y,multi_output=True)
        self.reshape(X) #compute self._XX
        self.y_ = y
        self.nn_ = define_model_all(shape = self.shape_,\
                                    n_feat=self.n_feat_, filter_size= self.filter_size_,\
                                    nhid1=self.nhid1_, nhid2 = self.nhid2_,\
                                    pool_size = self.pool_size_,lr=self.lr_)

  
        self.history_ = self.nn_.fit(self.XX_,self.y_,batch_size=self.batch_size_,\
                                  nb_epoch=self.nb_epoch_,\
                                  validation_split=self.validation_split_,verbose=0)
    
        return self

    def predict(self,X):
  #      ckeck_is_fitted(self,['nn_','history_'])
        X = check_array(X)
        self.reshape(X)
        return self.nn_.predict(self.XX_)


    def get_params(self,deep=True):
        return {name:getattr(self,name) for name in self.paramset_}

    def set_params(self,**parameters):
        #TODO Test if parameters are correct ?
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self
