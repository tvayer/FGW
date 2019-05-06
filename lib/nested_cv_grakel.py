# -*- coding: utf-8 -*-

from grakel import datasets
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
import itertools
import ast
import os
import argparse
from utils import str2bool,setup_logger,unique_repr,split_train_test
from gk_classifier import GK_classifier

class SearchError(Exception):
    pass

#%%

def explode_tuned_parameters(tuned_params):
    tuned_parameters2=tuned_params[0]
    varNames = sorted(tuned_parameters2)
    combinations = [dict(zip(varNames, prod)) for prod in itertools.product(*(tuned_parameters2[varName] for varName in varNames))]

    return combinations
    
def filter_dict(old_dict,your_keys):
    return { your_key: old_dict[your_key] for your_key in your_keys }

def filter_all_params(allparams):
    
    filtered_all_params=[]
    filtered_all_params.append(allparams[0])
    
    filtre=set({'kernel_params', 'normalize'}).intersection(set(allparams[0].keys()))
    filter_base=filter_dict(allparams[0],filtre)
    
    allreadydone=[]
    allreadydone.append(filter_base)
    
    for param in allparams:
        filter_param=filter_dict(param,filtre)
        if  filter_param not in allreadydone:
            allreadydone.append(filter_param)
            filtered_all_params.append(param)
    return filtered_all_params
    

#%%

def nested_cv(G,y,tuned_parameters,logging,n_iter=10,n_inner=10,verbose=1):

    logging.info('############ Begin nested CV ############')
    logging.info('Inner : '+str(n_inner))
    logging.info('Outer : '+str(n_iter))
    logging.info('params : '+str(tuned_parameters))


    outer_score=[]
    allparams=explode_tuned_parameters(tuned_parameters)
    
    all_params_filtered=filter_all_params(allparams)
    
    logging.info('Begin precomputing all Gram matrices')
    logging.info(str(len(all_params_filtered))+' matrices to fit...')
    
    dict_of_gram={}
    l=0
    for params in all_params_filtered:
        clf=GK_classifier(**params)
        K=clf.gk.fit_transform(G)
        dict_of_gram[unique_repr(clf.get_kernel_params(),'not_normal')]=K
        l+=1
        if l%10==0 and verbose>1:
            print('Done params : ',l)
    logging.info('...Done')
    
    clf=GK_classifier(precomputed=True)
     
    for i in range(n_iter):
        k_fold=StratifiedKFold(n_splits=n_inner,random_state=i)

        G_train,y_train,idx_train,G_test,y_test,idx_test=split_train_test(list(zip(G, list(y))),ratio=0.9, seed=i)        
        

        acc_inner_dict={} 
        best_inner_dict={}
        for param in allparams:
            acc_inner_dict[repr(param)]=[]    
            
        # fait un découpage de 9/10 du train 
        for idx_subtrain, idx_valid in k_fold.split(G_train,y_train):
            true_idx_subtrain=[idx_train[i] for i in idx_subtrain]
            true_idx_valid=[idx_train[i] for i in idx_valid]

            x_subtrain = [G[i] for i in true_idx_subtrain]
            y_subtrain = [y[i] for i in true_idx_subtrain]
            x_valid=[G[i] for i in true_idx_valid]
            y_valid=[y[i] for i in true_idx_valid]
                      
            # pour chaque parametre fit et test sur un subtrain subtest et inscrit le score    
            for param in allparams:
                # Initialise an SVM and fit.
                clf.set_params(**param)
                
                if unique_repr(clf.get_kernel_params(),'not_normal') in dict_of_gram:
                    K=dict_of_gram[unique_repr(clf.get_kernel_params(),'not_normal')]                                               
                    K_subtrain=K[np.ix_(true_idx_subtrain,true_idx_subtrain)]
                
                    # Fit on the train Kernel
                    clf.fit(K_subtrain, y_subtrain)
                
                    # Predict and test.
                    K_valid=K[np.ix_(true_idx_valid,true_idx_subtrain)]
                    y_pred = clf.predict(K_valid)
                    
                    # Calculate accuracy of classification.
                    ac_score=accuracy_score(y_valid, y_pred)
                    if verbose >1:
                        logging.info('----------------------------------------')
                        logging.info('----------------------------------------')
                        logging.info(' kernel params : '+str(clf.gk.get_params()))
                        logging.info(' svm params : '+str(clf.svc.get_params()))
                        logging.info(' score : '+str(ac_score))
                    
                    acc_inner_dict[repr(param)].append(ac_score)
                else :
                    print('dict_of_gram : ',dict_of_gram)
                    raise SearchError('not in dict_of_gram : \n param filtered : '+str(unique_repr(clf.get_kernel_params())))
                    
                
            logging.info('############ All params Done for one inner cut ############')
        
        logging.info('############ One inner CV Done ############')
              
        # Trouve les meilleurs params sur le inner CV
        for key,value in acc_inner_dict.items():
            best_inner_dict[key]=np.mean(acc_inner_dict[key])
                
        param_best=ast.literal_eval(max(best_inner_dict,key=best_inner_dict.get))
        logging.info('Best params : '+str(repr(param_best)))
        logging.info('Best inner score : '+str(max(list(best_inner_dict.values()))))
        
        clf.set_params(**param_best)
        
        K=dict_of_gram[unique_repr(clf.get_kernel_params(),'not_normal')]
        K_train=K[np.ix_(idx_train,idx_train)]
        K_test=K[np.ix_(idx_test,idx_train)]

        
        
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        
        ac_score_outer=accuracy_score(y_test, y_pred)
        outer_score.append(ac_score_outer)

        logging.info('Outer accuracy '+str(ac_score_outer))
        logging.info('############ One outer Done ############')
              
    logging.info('Nested mean score '+str(np.mean(outer_score)))
    logging.info('Nested std score '+str(np.std(outer_score)))
    
    
    

    
#%%    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nested GRAKEL')
    parser.add_argument('-m','--model_name', type=str,help='the name of the model',choices=['wl','sp','gk','rw','hopper','propa'],required=True)
    parser.add_argument('-dn','--dataset_name', type=str,help='the name of the dataset',choices=['mutag','ptc','nci1','imdb-b','imdb-m','enzymes','protein','bzr','cox2','synthetic','aids','cuneiform'],required=True)
    parser.add_argument('-o','--optionnal_name', nargs='?',default="",help='optionnal name to add for the log file')
    parser.add_argument('-r','--result_directory', nargs='?',default="", type=str,help='the path to the directory where to write to')
    parser.add_argument('-ni','--n_inner', nargs='?', default=10, type=int,help='the number of folds in the inner cv')
    parser.add_argument('-no','--n_outer', nargs='?', default=10, type=int,help='the number of folds in the outer cv')
    parser.add_argument('-C','--Csvm', nargs='?',default=-1, type=float,help='C parameter in Linear SVM. If not specified cross validated')
    parser.add_argument('-hwl','--hwl', nargs='?',default=-1, type=int,help='h parameter in WL. If not specified cross validated')
    parser.add_argument('-k','--gkk', nargs='?',default=-1, type=int,help='k parameter in GK. If not specified cross validated')
    parser.add_argument('-l','--lambd', nargs='?',default=-1, type=int,help='lambda parameter in RW. If not specified cross validated')
    parser.add_argument('-n','--newlib',nargs='?',help='wether to use newlib',type=str2bool,default=True)
    parser.add_argument('-v','--verbose', nargs='?', default=1, type=int,help='verbose')
    parser.add_argument('-test','--test',nargs='?',help='wether test version',type=str2bool,default=False)
    parser.add_argument('-lab','--use_lab',nargs='?',help='wether to use node labels',type=str2bool,default=True)


    args = parser.parse_args()
    
    name=args.model_name+'_'+'_'+args.dataset_name+'_'
    if args.hwl !=-1:
        name=name+'wl_'+str(args.hwl)
    name=name+'_'+args.optionnal_name
    try:
        if not os.path.exists(args.result_directory):
            os.makedirs(args.result_directory)
    except OSError:
        raise

    logger = setup_logger('outer_logger',args.result_directory+name+'_outer.log')
    logger.info('Let the Outer CV Begin for '+str(name))
    logger.info('n_outer : '+str(args.n_outer))
    logger.info('n_inner : '+str(args.n_inner))
    
    if args.Csvm !=-1:
        Clist=[args.Csvm]
    else:
        Clist=list(np.logspace(-5,5,15))   
    if args.hwl !=-1:
        hlist=[args.hwl]
    else:
        hlist=list(range(1,10))
    if args.gkk !=-1:
        gkklist=[args.gkk]
    else:
        gkklist=[3,4,5]
    if args.lambd !=-1:
        lambdlist=[args.lambd]
    else:
        lambdlist=[1e-2,1e-3,1e-4,1e-5,1e-6]

    if args.dataset_name=='mutag':
        dataset = datasets.fetch_dataset("MUTAG", verbose=False)
    if args.dataset_name=='ptc':
        dataset = datasets.fetch_dataset("PTC_MR", verbose=False)
    if args.dataset_name=='nci1':
        dataset = datasets.fetch_dataset("NCI1", verbose=False)
    if args.dataset_name=='imdb-b':
        dataset = datasets.fetch_dataset("IMDB-BINARY", verbose=False)
    if args.dataset_name=='imdb-m':
        dataset = datasets.fetch_dataset("IMDB-MULTI", verbose=False)
    if args.dataset_name=='enzymes':
        dataset = datasets.fetch_dataset("ENZYMES", verbose=False,prefer_attr_nodes=True)
    if args.dataset_name=='protein': 
        dataset = datasets.fetch_dataset("PROTEINS_full", verbose=False,prefer_attr_nodes=True)
    if args.dataset_name=='bzr': 
        dataset = datasets.fetch_dataset("BZR", verbose=False,prefer_attr_nodes=True)
    if args.dataset_name=='cox2': 
        dataset = datasets.fetch_dataset("COX2", verbose=False,prefer_attr_nodes=True)
    if args.dataset_name=='synthetic': 
        dataset = datasets.fetch_dataset("SYNTHETIC", verbose=False,prefer_attr_nodes=True) 
    if args.dataset_name=='aids': 
        dataset = datasets.fetch_dataset("AIDS", verbose=False,prefer_attr_nodes=True)  
    if args.dataset_name=='cuneiform': 
        dataset = datasets.fetch_dataset("Cuneiform", verbose=False,prefer_attr_nodes=True)
    G, y = dataset.data, dataset.target
    if args.test:
        G=G[1:800]
        y=y[1:800]
    
    if args.test:
        G=G[1:800]
        y=y[1:800]
    
    if args.model_name=='wl':
        tuned_parameters = [{'C':Clist,'kernel_params':[[{"name": "weisfeiler_lehman", "niter": x},
                                                         {"name": "subtree_wl"}] for x in hlist],'normalize':[True]}]
        nested_cv(G,y,tuned_parameters,logger,n_iter=args.n_outer,n_inner=args.n_inner,verbose=args.verbose)
    if args.model_name=='sp':
        tuned_parameters = [{'C':Clist,'kernel_params':[[{"name": "shortest_path","with_labels":args.use_lab}]],'normalize':[True]}] 
        nested_cv(G,y,tuned_parameters,logger,n_iter=args.n_outer,n_inner=args.n_inner,verbose=args.verbose)
    if args.model_name=='gk':        
        S=[{"delta":0.1,"epsilon":0.1},{"delta":0.05,"epsilon":0.1},{"delta":0.05,"epsilon":0.05},{"delta":0.1,"epsilon":0.05}]
        tuned_parameters = [{'C':Clist,'kernel_params':[[{"name": "graphlet_sampling","k":kx,"sampling":sx}] for kx,sx in itertools.product(gkklist,S)],'normalize':[True]}]
        nested_cv(G,y,tuned_parameters,logger,n_iter=args.n_outer,n_inner=args.n_inner,verbose=args.verbose)          
    if args.model_name=='rw':
        tuned_parameters = [{'C':Clist
                 ,'kernel_params':[[{"name": "random_walk","with_labels":args.use_lab,"lamda":l}] for l in lambdlist]
                ,'normalize':[True]}] 
        nested_cv(G,y,tuned_parameters,logger,n_iter=args.n_outer,n_inner=args.n_inner,verbose=args.verbose)
    if args.model_name=='hopper':
        tuned_parameters = [{'C':Clist
                 ,'kernel_params':[[{"name": "graph_hopper","kernel_type":'linear'}]]
                ,'normalize':[True]}] 
        nested_cv(G,y,tuned_parameters,logger,n_iter=args.n_outer,n_inner=args.n_inner,verbose=args.verbose)
    if args.model_name=='propa':
        tmaxlist=[1,3,5,8,10,15,20]
        tuned_parameters = [{'C':Clist
                 ,'kernel_params':[[{"name": "propagation","with_attributes":True,"t_max":x}] for x in tmaxlist]
                ,'normalize':[True]}] 
        nested_cv(G,y,tuned_parameters,logger,n_iter=args.n_outer,n_inner=args.n_inner,verbose=args.verbose)

        



    
    
    