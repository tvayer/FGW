import logging
from scipy.spatial.distance import cdist
import numpy as np
import argparse
import pickle
import os
import datetime,dateutil
import sys
import random 

def create_log_dir(FLAGS):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    log_dir = FLAGS.log_dir + "/" + str(sys.argv[0][:-3]) + "_" + timestamp 
    print(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save command line arguments
    with open(log_dir + "/hyperparameters_" + timestamp + ".csv", "w") as f:
        for arg in FLAGS.__dict__:
            f.write(arg + "," + str(FLAGS.__dict__[arg]) + "\n")

    return log_dir

def unique_repr(dictio,type_='normal'):
    """Compute a hashable unique representation of a list of dict with unashable values"""
    if 'normal':
        t = tuple((k, dictio[k]) for k in sorted(dictio.keys()))
    if 'not_normal':
        t=()
        for k in sorted(dictio.keys()):
            if not isinstance(dictio[k],list):
                t=t+((k, dictio[k]),)
            else: #suppose list of dict
                listechanged=[]
                for x in dictio[k]:
                    for k2 in sorted(x.keys()):
                        if not isinstance(x[k2],dict):
                            listechanged.append((k2,x[k2]))
                        else:
                            listechanged.append((k2,tuple((k3, x[k2][k3]) for k3 in sorted(x[k2].keys()))))
                tupletoadd=((k, tuple(listechanged)),)
                t=t+tupletoadd
    return t

def save_obj(obj, name,path='obj/' ):
    try:
        if not os.path.exists(path):
            print('Makedir')
            os.makedirs(path)
    except OSError:
        raise
    with open(path+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,path):
    with open(path + name, 'rb') as f:
        return pickle.load(f)

def indices_to_one_hot(number, nb_classes,label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""
    
    if number==label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]

def dist(x1, x2=None, metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2 using function scipy.spatial.distance.cdist
    Parameters
    ----------
    x1 : np.array (n1,d)
        matrix with n1 samples of size d
    x2 : np.array (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str, fun, optional
        name of the metric to be computed (full list in the doc of scipy),  If a string,
        the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns
    -------
    M : np.array (n1,n2)
        distance matrix computed with given metric
    """
    if x2 is None:
        x2 = x1

    return cdist(x1, x2, metric=metric)

def reshaper(x):
    x=np.array(x)
    try:
        a=x.shape[1]
        return x
    except IndexError:
        return x.reshape(-1,1)

def hamming_dist(x,y):
    #print('x',len(x[-1]))
    #print('y',len(y[-1]))
    return len([i for i, j in zip(x, y) if i != j])   

def allnan(v): #fonctionne juste pour les dict de tuples
    from math import isnan
    import numpy as np
    return np.all(np.array([isnan(k) for k in list(v)]))
def dict_argmax(d):
    l={k:v for k, v in d.items() if not allnan(v)}
    return max(l,key=l.get)
def dict_argmin(d):
    return min(d, key=d.get)

def read_files(mypath):
    from os import listdir
    from os.path import isfile, join

    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def per_section(it, is_delimiter=lambda x: x.isspace()):
    ret = []
    for line in it:
        if is_delimiter(line):
            if ret:
                yield ret  # OR  ''.join(ret)
                ret = []
        else:
            ret.append(line.rstrip())  # OR  ret.append(line)
    if ret:
        yield ret
        
def split_train_test(dataset,ratio=0.9, seed=None):
   idx_train = []
   X_train = []
   X_test = []
   random.seed(seed)
   for idx, val in random.sample(list(enumerate(dataset)),int(ratio*len(dataset))):
       idx_train.append(idx)
       X_train.append(val)
   idx_test=list(set(range(len(dataset))).difference(set(idx_train)))
   for idx in idx_test:
       X_test.append(dataset[idx])
   x_train,y_train=zip(*X_train)
   x_test,y_test=zip(*X_test)
   return np.array(x_train),np.array(y_train),np.array(idx_train),np.array(x_test),np.array(y_test),np.array(idx_test)    

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

