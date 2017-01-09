"""
This is a module to make a equivalent to scikit-learn estimator
with a deep neural network implemented in Keras"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y
from operator import mul
from functools import reduce
from datatools import define_model_all

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

    def __init__(self,shapef,n_feat=5,filter_size=3,nhid1=12,
                 nhid2=12,pool_size=(2,2),
                 batch_size=256,nb_epoch=50,validation_split=0.05):
        self.shapef_ = shapef
        self.n_feat_ = n_feat
        self.filter_size_ = filter_size
        self.nhid1_ = nhid1
        self.nhid2_ = nhid2
        self.pool_size_ = pool_size
        self.batch_size_ = batch_size
        self.nb_epoch_ = nb_epoch
        self.validation_split_ = validation_split

    def reshape(self,X):
        nfeature = reduce(mul,self.shapef_)
        assert(nfeature==X.shape[1]),'wrong size of X'
        self.XX_ = X.reshape(self.shape_)

    def fit(self,X,y):
        X, y = check_X_y(X,y,multi_output=True)
        self.shape_ = tuple([X.shape[0]])+tuple(self.shapef_) 
        self.reshape(X) #compute self._XX
        self.y_ = y
        self.nn_ = define_model_all(shape = self.shape_,\
                                    n_feat=self.n_feat_, filter_size= self.filter_size_,\
                                    nhid1=self.nhid1_, nhid2 = self.nhid2_,\
                                    pool_size = self.pool_size_)

  
        self.history_ = self.nn_.fit(self.XX_,self.y_,batch_size=self.batch_size_,\
                                  nb_epoch=self.nb_epoch_,\
                                  validation_split=self.validation_split_)
    
        return self

    def predict(self,X):
  #      ckeck_is_fitted(self,['nn_','history_'])
        
        self.reshape(X.values)
        return self.nn_.predict(self.XX_)

#TODO : check one learning
#TODO : getparam, setparam
