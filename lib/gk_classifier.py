#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:02:05 2018

@author: Titouan
"""
from sklearn.svm import SVC
from grakel import GraphKernel
import numpy as np

class BadInputError(Exception):
    pass
class GKNotDefinedError(Exception):
    pass

class GK_classifier():
    def __init__(self,C=1,precomputed=False,normalize=True,**params):
        self.C=C
        self.gk=self._create_gk_from_dict(**params)
        if self.gk is not None:
            self.gk.set_params(normalize=normalize)
        self.svc=SVC(kernel='precomputed', C=self.C)
        self.precomputed=precomputed
        self.normalize=normalize
        
    def __eq__(self, another):
        
        bool1=hasattr(another, 'gk')
        bool2=hasattr(another, 'svc')
        bool3=hasattr(another, 'precomputed')
        bool4=hasattr(another, 'normalize')
        bool5=self._check_eq_kernel_params(self.gk.kernel,another.gk.kernel)
        bool6=self.C==another.C
        bool7=self.precomputed==another.precomputed
        bool8=self.normalize==another.normalize

        return np.all([bool1,bool2,bool3,bool4,bool5,bool6,bool7,bool8])

    def __hash__(self):
        return hash(repr(self.get_params()))
        
    def fit(self,X,y=None):
        if self.gk is None:
            raise GKNotDefinedError
        if not self.precomputed:
            K=self.gk.fit_transform(X)
            self.svc.fit(K,y)
        else:
            self.svc.fit(X,y)
        
    def predict(self,X):
        if self.gk is None:
            raise GKNotDefinedError
        if not self.precomputed:
            K=self.gk.transform(X)
            return self.svc.predict(K)
        else:
            return self.svc.predict(X)
    
    def _create_gk_from_dict(self,**params): 
        ''' params={'kernel_params':[{'name':'shortest_path','with_labels':True}]}
        clf=GK_classifier(C=2,normalize=False,**params)''' 

        if 'kernel_params' not in params:
            print('Warning : no GK defined because kernel_params not in params')
            print('params : ',params)
            return None
        elif not isinstance(params['kernel_params'],list):
            raise BadInputError('Input[kernel_params] should be a list')
        else:
            return GraphKernel(params['kernel_params'])

    def set_one_param(self,dicto,key):
        if key in dicto:
            setattr(self, key, dicto[key])
            
    def set_params(self, **parameters):
        self.set_one_param(parameters,"C")
        self.set_one_param(parameters,"normalize")
        self.gk=self._create_gk_from_dict(**parameters)
        self.gk.set_params(normalize=self.normalize)
        self.svc=SVC(kernel='precomputed', C=self.C)
        return self
    
    def get_params(self, deep=True):
        if self.gk is None:
            return {"C":self.C,"kernel":None,"normalize":None}
        else:
            return {"C":self.C,"kernel":self.gk.kernel,"normalize":self.gk.normalize}

    def get_kernel_params(self, deep=True):
        if self.gk is None:
            return {"kernel_params":None ,"normalize":None}
        else:
            return {"kernel_params":self.gk.kernel #is a list
                    ,"normalize":self.gk.normalize}
                
    def _check_eq_kernel_params(self,param1,param2):
        if len(param1)!=len(param2):
            return False
        else:
            allTrue=[]
            for i in range(len(param1)):
                allTrue.append(param1[i] in param2) #may not be in same order
            return np.all(allTrue)
            



