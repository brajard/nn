"""
This is a module to make a equivalent to scikit-learn estimator
with a deep neural network implemented in Keras"""
import datatools
from importlib import reload
reload(datatools)

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from operator import mul
from functools import reduce
from datatools import define_model_all,  define_model_lstm, define_model_Dense, define_model_Conv
from keras.callbacks import EarlyStopping

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

    def __init__(self,shapef_=None,n_feat_in_=5,filter_size_in_=3,n_feat_out_=5,filter_size_out_=3,nhid1_=12,
                 nhid2_=12,pool_size_=(2,2),lr_=0.001,
                 batch_size_=256,nb_epoch_=50,validation_split_=0.05,init_=0,network_type_='all',earlystop_=True):
        if not shapef_:
            raise ValueError('shapef_ argument not set')
        self.shapef_ = shapef_
        self.n_feat_in_ = n_feat_in_
        self.filter_size_in_ = filter_size_in_
        self.n_feat_out_ = n_feat_out_
        self.filter_size_out_ = filter_size_out_
        self.nhid1_ = nhid1_
        self.nhid2_ = nhid2_
        self.pool_size_ = pool_size_
        self.batch_size_ = batch_size_
        self.nb_epoch_ = nb_epoch_
        self.validation_split_ = validation_split_
        print("validation_split_(init):",self.validation_split_)
        self.init_=init_
        self.lr_=lr_
        print("earlystop:",earlystop_)
        self.earlystop_=earlystop_
        print("self.earlstop_(init):",self.earlystop_)
        self.network_type_=network_type_
        self.paramset_ = {'shapef_','n_feat_in_','filter_size_in_','n_feat_out_','filter_size_out_','nhid1_',
                          'nhid2_','init_','pool_size_',
                          'batch_size_','nb_epoch_','validation_split_','earlystop_','lr_','network_type_'}

    def reshape(self,X):
        self.set_shape(X.shape)
        self.XX_ = X.reshape(self.shape_)
        
    def set_shape(self,shapex=None):
        nfeature = reduce(mul,self.shapef_)
        if shapex is None:
            shapex = tuple([1])+tuple([nfeature])
        assert(nfeature==shapex[1]),'wrong size of X'
        if self.network_type_ == 'all' or self.network_type_ == 'lstm':
            self.shape_ = tuple([shapex[0]])+tuple(self.shapef_) 
        elif self.network_type_ == 'dense' or self.network_type_ == 'conv':
            self.shape_ = tuple([shapex[0]])+tuple([self.shapef_[0]*self.shapef_[1]])+tuple(self.shapef_[2:])
        else:
            raise ValueError('not a valid model type')

    def set_model(self):
        if not hasattr(self,'shape_'):
            raise NameError('shape_ attribute is not defined')
        if (self.network_type_ == 'all' or self.network_type_ == 'lstm') and not len(self.shape_) == 5 :
            raise ValueError('invalid shape for model type')
        if ( self.network_type_ == 'dense' or self.network_type_ == 'conv') and not len(self.shape_) == 4:
            raise ValueError('invalid shape for model type')
        
        if self.network_type_ == 'all':
            self.nn_ = define_model_all(shape = self.shape_,
                                        n_feat_in=self.n_feat_in_, n_feat_out=self.n_feat_out_,
                                        filter_size_in= self.filter_size_in_,
                                        filter_size_out= self.filter_size_out_,
                                        nhid1=self.nhid1_, nhid2 = self.nhid2_,
                                        pool_size = self.pool_size_,lr=self.lr_)

        elif self.network_type_ == 'lstm':
            self.nn_ = define_model_lstm(shape = self.shape_,
                                    nhid1=self.nhid1_, nhid2 = self.nhid2_,
                                    lr=self.lr_)

        elif self.network_type_ == 'dense':
            self.nn_ = define_model_Dense(shape = self.shape_,
                                    nhid1=self.nhid1_, nhid2 = self.nhid2_,
                                    lr=self.lr_)

        elif self.network_type_ == 'conv':
            self.nn_ = define_model_Conv(shape = self.shape_,
                                         n_feat_in=self.n_feat_in_, n_feat_out=self.n_feat_out_,
                                         filter_size_in= self.filter_size_in_,
                                         filter_size_out= self.filter_size_out_,
                                         nhid=self.nhid1_,
                                         pool_size = self.pool_size_,lr=self.lr_)
        else:
            raise ValueError('not a valid model type')

        return self.nn_
            

    def fit(self,X,y,verbose=0):
        X, y = check_X_y(X,y,multi_output=True)
        self.reshape(X) #compute self.XX_
        self.y_ = y
        self.set_model()
        callbacks = None
        print("self.earlstop_(fit):",self.earlystop_)

        if self.earlystop_:
            callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]
        print("callback",callbacks)
        self.history_ = self.nn_.fit(self.XX_,self.y_,batch_size=self.batch_size_,\
                                  nb_epoch=self.nb_epoch_,\
                                  callbacks=callbacks,\
                                  validation_split=self.validation_split_,verbose=verbose)
    
        return self

    def predict(self,X):
  #      ckeck_is_fitted(self,['nn_','history_'])
        X = check_array(X)
        self.reshape(X)
        return self.nn_.predict(self.XX_)


    def get_params(self,deep=True):
        return {name:getattr(self,name) for name in self.paramset_}

    def set_params(self,**parameters):
        for parameter, value in parameters.items():
            if not parameter in self.paramset_:
                raise ValueError ("Parameter '"+parameter+"' is not in the admissible list")
            setattr(self,parameter,value)
        return self
