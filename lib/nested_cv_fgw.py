# -*- coding: utf-8 -*-

"""
Compute the nested Cross Validation used in the paper for fgw.
"""

from sklearn.model_selection import StratifiedKFold
import itertools
import os
import numpy as np
import ast
from utils import split_train_test,unique_repr,create_log_dir
from sklearn.metrics import accuracy_score
from custom_svc import Graph_FGW_SVC_Classifier
import time 
import utils 
import argparse
from data_loader import load_local_data

class AlphaMustBeDefinedError(Exception):
    pass

def explode_tuned_parameters(tuned_params):
    tuned_parameters2=tuned_params[0]
    varNames = sorted(tuned_parameters2)
    combinations = [dict(zip(varNames, prod)) for prod in itertools.product(*(tuned_parameters2[varName] for varName in varNames))]

    return combinations

#%%

def filter_dict(old_dict,your_keys):
    return { your_key: old_dict[your_key] for your_key in your_keys }
    
def filter_all_params(allparams,filtre_key):
    filtered_all_params=[]
    filtre=set(filtre_key).intersection(set(allparams[0].keys()))
    for param in allparams:
        filter_param=filter_dict(param,filtre)
        if filter_param not in filtered_all_params:
            filtered_all_params.append(filter_param)
    return filtered_all_params
    

#%%

    
def nested_fgw(X,y,tuned_parameters,dataset_name,logging,path=None,n_inner=10,n_iter=10,verbose=1,optionnal=""):

    """ Compute the nested cross-validation         
        Parameters
        ----------
        X : array of Graph objects
        y : array of classes of each graph
        tuned_parameters : a list of dict 
                           Parameters to cross validate. Identical ass sklearn for GridSearch
        dataset_name : string 
                       name of the dataset. Used only to check the right dataset in the precalculated disances folder
        logging : a logging object
                  Used to write the log. Can be instantiate via utils.setup_logger
        path : string
               Path to the precalculated FGW distances. If not specified all distances are recalculated. 
               If specified it checks amoung all precalculated distances the ones that correspond to the cross-validated parameters
        n_inner : integer
                  The number of inner folds in the nested cross validation
        n_iter : integer
                 The number of outer folds in the nested cross validation
        optionnal : string
                    A optionnal name to add to the name of the log file
        Returns
        -------
        Writes the results in the log file
    """

    logging.info('############ Begin nested CV ############')
    logging.info('Inner : '+str(n_inner))
    logging.info('Outer : '+str(n_iter))
    logging.info('params : '+str(tuned_parameters))
    
    X=np.array(X)
    y=np.array(y)
    
    outer_score=[]
    allparams=explode_tuned_parameters(tuned_parameters)
    
    filtre=set(Graph_FGW_SVC_Classifier().get_distances_params().keys())
    all_params_filtered=filter_all_params(allparams,filtre)
    
    logging.info('Begin precomputing all distances matrices')
    logging.info(str(len(all_params_filtered))+' matrices to fit...')
    # Get the distances of calculates them
    dict_of_all_distances={}
    l=0
    for params in all_params_filtered:
        clf=Graph_FGW_SVC_Classifier(**params)
        if path is None:
            if verbose>1:
                print('Path is None : we precalculate distances but we are not saving them')
            clf.compute_all_distance(np.array(X),np.array(X))
            dict_of_all_distances[unique_repr(clf.get_distances_params())]=clf.D
        else:
            if optionnal!="":
                name=dataset_name+optionnal
            else:
                name=dataset_name
            if name+'.pkl' in os.listdir(path):
                if verbose>1:
                    print('Load dict')
                d=utils.load_obj(name+'.pkl',path=path) #load the distances of the given dataset
            else:
                if verbose>1:
                    print('Create empty dict')
                d={}
            if unique_repr(clf.get_distances_params()) in d: #if cv param are there we get the distance
                D=d[unique_repr(clf.get_distances_params())]
                dict_of_all_distances[unique_repr(clf.get_distances_params())]=D/np.max(D)
            else:
                if verbose >1:
                    print('Recalculate distance')
                clf.compute_all_distance(np.array(X),np.array(X))
                dict_of_all_distances[unique_repr(clf.get_distances_params())]=clf.D/np.max(clf.D)
                d[unique_repr(clf.get_distances_params())]=clf.D
                utils.save_obj(d,name,path=path) #save the distances
            logging.info('One distance done')                     
        l+=1
        if l%10==0 and verbose>1:
            print('Done params : ',l)
    logging.info('...Done')
            
    for i in range(n_iter): # do the nested CV
        k_fold=StratifiedKFold(n_splits=n_inner,random_state=i)
        G_train,y_train,idx_train,G_test,y_test,idx_test=split_train_test(list(zip(X, list(y))),ratio=0.9, seed=i)        

        acc_inner_dict={} 
        best_inner_dict={}
        for param in allparams:
            acc_inner_dict[repr(param)]=[]    
            
        for idx_subtrain, idx_valid in k_fold.split(G_train,y_train):
            true_idx_subtrain=[idx_train[i] for i in idx_subtrain]
            true_idx_valid=[idx_train[i] for i in idx_valid]

            x_subtrain = np.array([X[i] for i in true_idx_subtrain])
            y_subtrain = np.array([y[i] for i in true_idx_subtrain])
            x_valid=np.array([X[i] for i in true_idx_valid])
            y_valid=np.array([y[i] for i in true_idx_valid])
                      
            # For all parameter fit on subrain and test on subtest    
            for param in allparams:
                # Initialise an SVM and fit.
                clf = Graph_FGW_SVC_Classifier()
                clf.set_params(**param)
                                
                # Fit on the train Kernel
                                        
                if unique_repr(clf.get_distances_params()) in dict_of_all_distances:
                    
                    if verbose>2:
                        print('--------------------------------------------------------')
                        print('Params all : ', str(unique_repr(clf.get_params())))  
                        print('Distance pram : ', str(unique_repr(clf.get_distances_params())))
                    
                    D=dict_of_all_distances[unique_repr(clf.get_distances_params())]
                    st=time.time() 
                    clf.fit(x_subtrain,y_subtrain,matrix=D[np.ix_(true_idx_subtrain,true_idx_subtrain)])
                        
                    # Predict and test.
                    y_pred = clf.predict(x_valid,matrix=D[np.ix_(true_idx_valid,true_idx_subtrain)])
                    ed=time.time()
                    
                    # Calculate accuracy of classification.
                    ac_score=accuracy_score(y_valid.reshape(-1,1), y_pred.reshape(-1,1))
                    if verbose>2:
                        print('Done in : ',ed-st)
                        print('Accuracy score on inner : '+str(ac_score)) 
                        print('--------------------------------------------------------')
                    acc_inner_dict[repr(param)].append(ac_score)
                    
                                    
            logging.info('############ All params Done for one inner cut ############')

        
        logging.info('############ One inner CV Done ############')
              
        # Fin best params in the inner
        for key,value in acc_inner_dict.items():
            best_inner_dict[key]=np.mean(acc_inner_dict[key])
                
        param_best=ast.literal_eval(max(best_inner_dict,key=best_inner_dict.get))
        logging.info('Best params : '+str(repr(param_best)))
        logging.info('Best inner score : '+str(max(list(best_inner_dict.values()))))

        clf = Graph_FGW_SVC_Classifier()
        clf.set_params(**param_best)  
        
        D=dict_of_all_distances[unique_repr(clf.get_distances_params())]
                    
        clf.fit(G_train, y_train,matrix=D[np.ix_(idx_train,idx_train)])
        
        y_pred = clf.predict(G_test,matrix=D[np.ix_(idx_test,idx_train)])
        
        ac_score_outer=accuracy_score(y_test.reshape(-1,1), y_pred.reshape(-1,1))
        outer_score.append(ac_score_outer)

        logging.info('Outer accuracy '+str(ac_score_outer))
        logging.info('############ One outer Done ############')
              
    logging.info('Nested mean score '+str(np.mean(outer_score)))
    logging.info('Nested std score '+str(np.std(outer_score)))

    
#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nested CV for fgw')
    parser.add_argument('-dn','--dataset_name', type=str,help='the name of the dataset',choices=['mutag','ptc','nci1','imdb-b','imdb-m','enzymes','protein','protein_notfull','bzr','cox2','synthetic','aids','cuneiform'],required=True)
    parser.add_argument('-d','--data_path',type=str,help='the path to the data',required=True)
    parser.add_argument('-ni','--n_inner', nargs='?', default=10, type=int,help='the number of folds in the inner cv')
    parser.add_argument('-no','--n_outer', nargs='?', default=10, type=int,help='the number of folds in the outer cv')
    parser.add_argument('-dist','--distances_path', nargs='?',help='the path to the precalculated distances for fgw ')
    parser.add_argument('-o','--optionnal_name', nargs='?',default="",help='optionnal name to add for the log file')
    parser.add_argument('-r','--log_dir', nargs='?',default="", type=str,help='the path to the directory where to write to')
    parser.add_argument('-wl','--wl_feature',nargs='?',type=int,help='Use the Weisfeler Lehman features if wl>0 ',default=0)
    parser.add_argument('-at','--attributes',nargs='?',help='wether to use vector attributes of the graph',type=utils.str2bool,default=True)
    parser.add_argument('-fea','--feature_metric',type=str,choices=['euclidean','sqeuclidean','dirac','hamming_dist'],help='the metric to use for the features',required=True)
    parser.add_argument('-st','--structure_metric',type=str,choices=['shortest_path','weighted_shortest_path','harmonic_distance','adjency','square_shortest_path'],help='the metric to use for the structures',required=True)
    parser.add_argument('-C','--Csvm', nargs='?',default=-1, type=float,help='C parameter in Linear SVM. If not specified cross validated')
    parser.add_argument('-g','--gamma', nargs='?',default=-1, type=float,help='Gamma parameter in Gaussian SVM. If not specified cross validated')
    parser.add_argument('-test','--test',nargs='?',help='wether to use a test version',type=utils.str2bool,default=False)
    parser.add_argument('-v','--verbose', nargs='?', default=1, type=int,help='verbose')
    parser.add_argument('-am','--amijo',nargs='?',help='wether to use amijo linesearch',type=utils.str2bool,default=True)
    parser.add_argument('-a','--alpha', nargs='+',type=float, help='Alphas to cross validate. Ignored if cva is true',default=-8000)
    parser.add_argument('-cva','--automatic_cv_alpha',nargs='?',help='wether to use a predifined CV grid for alpha. 15 alphas are tested',type=utils.str2bool,default=False)

    
    args = parser.parse_args()
    data_path=args.data_path
    
    if args.alpha==-8000 and not args.automatic_cv_alpha:
        raise AlphaMustBeDefinedError('You must set alpha via -a or use automatic grid via -cva')
    
    
    name='fgw'+'_'+args.dataset_name+'_feature_metric_'+args.feature_metric+'_structure_metric_'+args.structure_metric
    if args.wl_feature>0:
        name=name+'_wl_'+str(args.wl_feature)
    name=name+args.optionnal_name
    
    try:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    except OSError:
        raise

    log_dir=create_log_dir(args)
  
    if args.Csvm !=-1:
        Clist=[args.Csvm]
    else:
        Clist=list(np.logspace(-4,4,15))
        
    if args.automatic_cv_alpha :
        nb=15
        N=int(15/3)
        a=np.logspace(-6,-1,N)
        c=1-a
        b=np.array(list(set(np.linspace(a[0],c[0],N)).difference(set((a[0],c[0])))))
        alphalist=np.concatenate((a,b,c))
        alpha_list=list(np.sort(np.append([0,1],alphalist)))
    else:
        alpha_list=args.alpha
    if args.gamma !=-1:
        gamma_list=[args.gamma]
    else:
        gamma_list=list([2**k for k in np.linspace(-10,10,15)])
    
    logger = utils.setup_logger('outer_logger', log_dir+'/'+name+'_outer.log')
    logger.info('Let the Outer CV Begin for '+str(name))
    logger.info('n_outer : '+str(args.n_outer))
    logger.info('n_inner : '+str(args.n_inner)) 
      
    X,y=load_local_data(data_path,args.dataset_name,attributes=args.attributes,wl=args.wl_feature)

    if args.test:
        X=X[1:50]
        y=y[1:50]

    tuned_parameters = [{'alpha':alpha_list,'C':Clist,'gamma':gamma_list,'features_metric':[args.feature_metric],
                         'method':[args.structure_metric],'wl':[args.wl_feature],'amijo':[args.amijo]}]
    
    nested_fgw(X,y
        ,tuned_parameters
        ,args.dataset_name
        ,logger
        ,args.distances_path
        ,n_inner=args.n_inner
        ,n_iter=args.n_outer
        ,verbose=args.verbose
        ,optionnal=str(args.optionnal_name))
    
    
    
    