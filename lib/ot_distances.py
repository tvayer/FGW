import ot
import FGW as fgw
import numpy as np
import time
from graph import NoAttrMatrix
from utils import hamming_dist

# A factoriser en utilisant une seule méthode à la place de tree_d etc : juste créer un objet qui a distance_matrix!!!!!
# TODO : add stab to all methods
class BadParameters(Exception):
    pass

class Wasserstein_distance():
    def __init__(self,features_metric='sqeuclidean'): #remplacer method par distance_method  
        self.features_metric=features_metric
        self.transp=None

    def reshaper(self,x):
        x=np.array(x)
        try:
            a=x.shape[1]
            return x
        except IndexError:
            return x.reshape(-1,1)

    def graph_d(self,graph1,graph2):

        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)
        x1=self.reshaper(graph1.all_matrix_attr())
        x2=self.reshaper(graph2.all_matrix_attr())

        if self.features_metric=='dirac':
            f=lambda x,y: x!=y
            M=ot.dist(x1,x2,metric=f)
        else:
            M=ot.dist(x1,x2,metric=self.features_metric) 
        if np.max(M)!=0:
            M= M/np.max(M)
        self.M=M

        transp = ot.emd(t1masses,t2masses, M)
        self.transp=transp

        return np.sum(transp*M)

    def get_tuning_params(self):
        return {"features_metric":self.features_metric}



class Fused_Gromov_Wasserstein_distance():


    def __init__(self,alpha=0.5,method='shortest_path',features_metric='sqeuclidean',max_iter=500,verbose=False,amijo=False): #remplacer method par distance_method  
        self.method=method
        self.max_iter=max_iter
        self.alpha=alpha
        self.features_metric=features_metric
        self.transp=None
        self.log=None
        self.verbose=verbose
        self.amijo=amijo
        #if alpha==0 or alpha==1:
        #    self.amijo=True

    def reshaper(self,x):
        try:
            a=x.shape[1]
            return x
        except IndexError:
            return x.reshape(-1,1)

    def calc_fgw(self,M,C1,C2,t1masses,t2masses):
        transpwgw,log= fgw.fgw_lp((1-self.alpha)*M,C1,C2,t1masses,t2masses,'square_loss',G0=None,alpha=self.alpha,verbose=self.verbose,amijo=self.amijo,log=True)      
        return transpwgw,log
        
    def graph_d(self,graph1,graph2):

        #print('fused')
        gofeature=True
        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        startstruct=time.time()
        C1=graph1.distance_matrix(method=self.method,algo='scipy')
        C2=graph2.distance_matrix(method=self.method,algo='scipy')
        end2=time.time()
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)
        try :
            x1=self.reshaper(graph1.all_matrix_attr())
            x2=self.reshaper(graph2.all_matrix_attr())
        except NoAttrMatrix:
            x1=None
            x2=None
            gofeature=False
        if gofeature : 
            if self.features_metric=='dirac':
                f=lambda x,y: x!=y
                M=ot.dist(x1,x2,metric=f)
            elif self.features_metric=='hamming_dist':
                f=lambda x,y: hamming_dist(x,y)
                M=ot.dist(x1,x2,metric=f)
            else:
                M=ot.dist(x1,x2,metric=self.features_metric)
            self.M=M
        else:
            M=np.zeros((C1.shape[0],C2.shape[0]))

        startdist=time.time()
        transpwgw,log=self.calc_fgw(M,C1,C2,t1masses,t2masses)
        enddist=time.time()

        enddist=time.time()
        log['struct_time']=(end2-startstruct)
        log['dist_time']=(enddist-startdist)
        self.transp=transpwgw
        self.log=log

        return log['loss'][::-1][0]


    

    def get_tuning_params(self):
        return {"method":self.method,"max_iter":self.max_iter,"alpha":self.alpha,
        "features_metric":self.features_metric,"amijo":self.amijo}
