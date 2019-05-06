# -*- coding: utf-8 -*-

from sklearn.svm import SVC
import numpy as np
from sklearn.base import TransformerMixin
from ot_distances import Fused_Gromov_Wasserstein_distance
import time
from optim import NonConvergenceError
from sklearn.exceptions import NotFittedError

class InfiniteException(Exception):
    pass

class NanErrorInDist(Exception):
    pass

"""
The following classes are used to create a SVM classifier over the FGW distance using the indefinite kernel e^{-\gamma*FGW}
"""

class GenericSVCClassifier(TransformerMixin):
    """ GenericSVCClassifier is a sklearn compatible class. 
    It computes a SVM classifier over a any type of data as long as a similarity measure is defined.
    More precisely if f is a similarity measure it computes a SVM on a precomputed similarity matrix K=exp{-gamma*f(x,y)} for all x,y
    
    Attributes
    ----------    
    similarity_measure : a method
               The similarity mesure between the points
    gamma : float
            The gamma parameter in the similarity matrix K=exp{-gamma*f(x,y)}
    D : ndarray
        The similarity matrix f(x,y)
    svc : the SVM classifier from sklearn
    C : float 
        The C parameter of the SVM

    """
    def __init__(self,similarity_measure,C=1,gamma=1,verbose=False,always_raise=False):
        self.similarity_measure = similarity_measure
        self.gamma=gamma
        self.C=C
        self.verbose=verbose
        self.D=None
        self.similarity_measure_time=[]
        self.infiniteExceptionOccuredInFit=False
        self.always_raise=always_raise
        self.svc=SVC(C=self.C,kernel="precomputed",verbose=self.verbose,max_iter=10000000)

    def compute_similarity(self,x,y):
        
        """ Compute the similarity between x and y using the similarity_measure
        Parameters
        ----------
        x : a abstract object
        y : a astract object
         Returns
        -------
        A float representative of the similarity
        """
        start=time.time()
        try:
            similarity=self.similarity_measure(x,y)
        except NonConvergenceError:
            print('NonConvergenceError for ',x.characterized(),y.characterized())
            similarity=np.nan
            if self.always_raise:
                raise NanErrorInDist
        if np.isnan(similarity) and self.always_raise:
            raise NanErrorInDist
        end=time.time()
        self.similarity_measure_time.append(end-start)
        return similarity

    def gram_matrix(self,X,Y,matrix=None,method='classic'):
        """ Compute the similarity matrix K=exp{-gamma*f(x,y)} with f the similarity measure 
        for all x,y in X and Y 
        Parameters
        ----------
        X : array of abstract object
        Y : array of abstract object
        matrix : ndarray, optionnal
                 If specified used to compute the similarity matrix instead of calculating all the similarities
        method : string
                 If equal to classic compute K=exp{-gamma*f(x,y)}, if equal to no_gaussian compute only f(x,y)
         Returns
        -------
        D : ndarray
            The gram matrix of all similarities K=exp{-gamma*f(x,y)} or f(x,y) if method='no_gaussian'
        """
        self.compute_all_distance(X,Y,matrix)
        if method=='classic':
            Z=np.exp(-self.gamma*(self.D))
            if not self.assert_all_finite(Z):
                raise InfiniteException('There is Nan')
            else:
                return Z
        if method=='no_gaussian':
            return self.D

    def fit(self,X,y=None,matrix=None):
        """ Fit the SVM classifier on the similarity matrix 
        Parameters
        ----------
        X : array of abstract object
        y : classes of all objects
        matrix : ndarray, optionnal
                 If specified used to compute the similarity matrix instead of calculating all the similarities
         Returns
        -------
        self
        """
        self.classes_ =np.array(y)
        self._fit_X=np.array(X)
        Gtrain = np.zeros((X.shape[0],X.shape[0]))
        start=time.time()
        try :
            Gtrain = self.gram_matrix(X,X,matrix,method='classic')
            self.svc.fit(Gtrain,self.classes_)
            if self.verbose:
                print('Time fit : ',time.time()-start)
        except InfiniteException:
            self.infiniteExceptionOccuredInFit=True
            print('InfiniteException : value error in fit because nan')
        return self

    def predict(self,X,matrix=None):
        """ Apply the SVM classifier on X
        Parameters
        ----------
        X : array of abstract object
        matrix : ndarray, optionnal
                 If specified used to compute the similarity matrix instead of calculating all the similarities
         Returns
        -------
        self
        """
        try :
            G=self.gram_matrix(X,self._fit_X,matrix,method='classic')
            preds=self.svc.predict(G)

        except InfiniteException:
            print('InfiniteException : Preds error because nan')
            preds=np.repeat(-10,len(X)) # Dirty trick so that preds are not None

        except NotFittedError:
            if self.infiniteExceptionOccuredInFit :
                print('NotFittedError : nan dans la gram de fit mais pas dans celle de test')
                preds=np.repeat(-10,len(X)) # Dirty trick so that preds are not None
            else:
                raise NotFittedError
        return preds

    def assert_all_finite(self,X):
        """Like assert_all_finite, but only for ndarray."""
        X = np.asanyarray(X)
        a=X.dtype.char in np.typecodes['AllFloat']
        b=np.isfinite(X.sum())
        c=np.isfinite(X).all()

        if (a and not b and not c):
            return False 
        else :
            return True

    def compute_all_distance(self,X,Y,matrix=None): 
        """ Compute all similarities f(x,y) for x,y in X and Y and f the similarity measure 
        Parameters
        ----------
        X : array of abstract object
        Y : array of abstract object
        matrix : ndarray, optionnal
                 If specified used to compute the similarity matrix instead of calculating all the similarities
         Returns
        -------
        None. Set the similarity matrix
        """
        if matrix is not None :
            self.D=matrix

        else:
            X=X.reshape(X.shape[0],) #idem
            Y=Y.reshape(Y.shape[0],) #idem

            if np.all(X==Y):
                D= np.zeros((X.shape[0], Y.shape[0]))
                H=np.zeros((X.shape[0], Y.shape[0]))
                for i, x1 in enumerate(X):
                    for j,x2 in enumerate(Y):
                        if j>=i:
                            dist=self.compute_similarity(x1, x2)
                            D[i, j] = dist
                np.fill_diagonal(H,np.diagonal(D))
                D=D+D.T-H
            else:
                D = np.zeros((X.shape[0], Y.shape[0]))
                for i, x1 in enumerate(X):
                    row=[self.compute_similarity(x1, x2) for j,x2 in enumerate(Y)]
                    D[i,:]=row
            D[np.abs(D)<=1e-15]=0 #threshold due to numerical precision

            self.D=D

       
    def set_one_param(self,dicto,key):
        if key in dicto:
            setattr(self, key, dicto[key])

    def get_params(self, deep=True):
        return {"similarity_measure":self.similarity_measure,"gamma":self.gamma,"C":self.C}

    def get_distances_params(self):
        return {"similarity_measure":self.similarity_measure}

    def set_params(self, **parameters):
        self.set_one_param(parameters,"similarity_measure")
        self.set_one_param(parameters,"C")
        self.set_one_param(parameters,"gamma")
        self.svc=SVC(C=self.C,kernel="precomputed",verbose=self.verbose,max_iter=10000000)
        return self


class Graph_FGW_SVC_Classifier(GenericSVCClassifier):
    """ Graph_FGW_SVC_Classifier is a generic class that inherit from GenericSVCClassifier. It uses the FGW as similarity measure
    
    Attributes
    ----------    
    gw : a Fused_Gromov_Wasserstein_distance instance
         The Fused_Gromov_Wasserstein_distance class for computing FGW
    alpha : float 
            The alpha parameter of FGW
    method : string
             The name of the method used to compute the structures matrices of the graphs. See Graph class
    max_iter : integer
               Number of iteration of the FW algorithm for the computation of FGW.
    features_metric : string
                      The name of the method used to compute the cost matrix between the features
                      For hamming_dist see experimental setup in [3]
    transp : ndarray, shape (ns,nt) 
           The transport matrix between the source distribution and the target distribution
    amijo : bool, optionnal
            If True the steps of the line-search is found via an amijo research. Else closed form is used.
            If there is convergence issues use False.
    wl : integer
         Parameter Weisfeler-Lehman attributes. See experimental setup of [3]
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.

    """
    def __init__(self,C=1,gamma=1,alpha=1,method='shortest_path',features_metric='sqeuclidean',verbose=False,always_raise=False,amijo=True,wl=0):
        
        
        self.gw=Fused_Gromov_Wasserstein_distance(alpha=alpha,method=method,features_metric=features_metric,amijo=amijo)
        similarity_measure=self.gw.graph_d
        
        GenericSVCClassifier.__init__(self,similarity_measure=similarity_measure,C=C,gamma=gamma,verbose=verbose)

        self.alpha=alpha
        self.features_metric=features_metric
        self.method=method
        self.wl=wl
        self.amijo=amijo
        GenericSVCClassifier.__init__(self,C=C,gamma=gamma,similarity_measure=similarity_measure,verbose=verbose,always_raise=always_raise)

    def fit(self,X,y=None,matrix=None): #avoid recalculating all structures matrices if they already exist
        self.classes_ = y
        self._fit_X = list(X.reshape(X.shape[0],)) 
        for x in self._fit_X :
            if x.C is None or x.name_struct_dist!=self.method:
                if self.verbose:
                    print('******************************************************')
                    print('Construction des matrices de structures')
                    if x.C is not None:
                        print('before ',x.name_struct_dist)
                        print('nw ',self.method)
                    else:
                        print('Because structure is None')
                    print('******************************************************')
                _=x.distance_matrix(method=self.method,force_recompute=True)
        super(Graph_FGW_SVC_Classifier,self).fit(X,y,matrix)

    def get_params(self, deep=True):
        return {"alpha":self.alpha
        ,"features_metric":self.features_metric
        ,"method":self.method
        ,"C":self.C
        ,"gamma":self.gamma
        ,"amijo":self.amijo
        ,"wl":self.wl
        }

    def set_params(self, **parameters):
        self.set_one_param(parameters,"alpha")
        self.set_one_param(parameters,"features_metric")
        self.set_one_param(parameters,"method")
        self.set_one_param(parameters,"C")
        self.set_one_param(parameters,"gamma")
        self.set_one_param(parameters,"amijo")    
        self.set_one_param(parameters,"wl")
        self.svc=SVC(C=self.C,kernel="precomputed",verbose=self.verbose,max_iter=10000000)

        gw2=Fused_Gromov_Wasserstein_distance(alpha=self.alpha,method=self.method,features_metric=self.features_metric,amijo=self.amijo)
        if self.gw.get_tuning_params()!=gw2.get_tuning_params(): #if not same tuning param recreate a new FGW object because the similarity measure changes
            self.gw=Fused_Gromov_Wasserstein_distance(alpha=self.alpha,method=self.method,features_metric=self.features_metric,amijo=self.amijo)
            self.similarity_measure=self.gw.graph_d

        return self

    def get_distances_params(self):
        dall = {}
        dall.update(self.gw.get_tuning_params())
        dall.update({'wl':self.wl})
        return dall

