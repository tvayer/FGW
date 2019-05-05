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


class GenericSVCClassifier(TransformerMixin):
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

    def gram_matrix(self,X,Y,matrix=None,y=None,method='classic'):
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

    def get_start_end_indexes(self,n_support_):
        start = [sum(n_support_[:i]) for i in range(len(n_support_))]
        end = [start[i] + n_support_[i] for i in range(len(n_support_))]
        return start,end

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

    def compute_all_distance(self,X,Y,matrix=None): # Il faut stocker ce kernel en dessous

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

    def __init__(self,C=1,gamma=1,alpha=1,method='shortest_path',features_metric='sqeuclidean',verbose=False,always_raise=False,amijo=False,wl=0):
        
        
        self.gw=Fused_Gromov_Wasserstein_distance(alpha=alpha,method=method,features_metric=features_metric,amijo=amijo)
        similarity_measure=self.gw.graph_d
        
        GenericSVCClassifier.__init__(self,similarity_measure=similarity_measure,C=C,gamma=gamma,verbose=verbose)

        self.alpha=alpha
        self.features_metric=features_metric
        self.method=method
        self.wl=wl
        self.amijo=amijo
        GenericSVCClassifier.__init__(self,C=C,gamma=gamma,similarity_measure=similarity_measure,verbose=verbose,always_raise=always_raise)

    def fit(self,X,y=None,matrix=None):
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
        if self.gw.get_tuning_params()!=gw2.get_tuning_params():
            self.gw=Fused_Gromov_Wasserstein_distance(alpha=self.alpha,method=self.method,features_metric=self.features_metric,amijo=self.amijo)
            self.similarity_measure=self.gw.graph_d

        return self

    def get_distances_params(self):
        dall = {}
        dall.update(self.gw.get_tuning_params())
        dall.update({'wl':self.wl})
        return dall

