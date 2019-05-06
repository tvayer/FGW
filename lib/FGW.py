# -*- coding: utf-8 -*-

import numpy as np
import ot
import optim
from utils import dist,reshaper
from bregman import sinkhorn_scaling
from scipy import stats
from scipy.sparse import random

class StopError(Exception):
    pass

def init_matrix(C1,C2,p,q,loss_fun='square_loss'):
    """ Return loss matrices and tensors for Gromov-Wasserstein fast computation
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
    function as the loss function of Gromow-Wasserstein discrepancy.
    The matrices are computed as described in Proposition 1 in [1]
    Where :
        * C1 : Metric cost matrix in the source space
        * C2 : Metric cost matrix in the target space
        * T : A coupling between those two spaces
    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=(a^2)
            * f2(b)=(b^2)
            * h1(a)=a
            * h2(b)=2b
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
         Coupling between source and target spaces
    p : ndarray, shape (ns,)
    Returns
    -------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
            
    if loss_fun == 'square_loss':
        def f1(a):
            return a**2 

        def f2(b):
            return b**2

        def h1(a):
            return a

        def h2(b):
            return 2*b

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC=constC1+constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC,hC1,hC2

def tensor_product(constC,hC1,hC2,T):

    """ Return the tensor for Gromov-Wasserstein fast computation
    The tensor is computed as described in Proposition 1 Eq. (6) in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
    
    A=-np.dot(hC1, T).dot(hC2.T)
    tens = constC+A

    return tens

def gwloss(constC,hC1,hC2,T):

    """ Return the Loss for Gromov-Wasserstein
    The loss is computed as described in Proposition 1 Eq. (6) in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    loss : float
           Gromov Wasserstein loss
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    tens=tensor_product(constC,hC1,hC2,T) 
              
    return np.sum(tens*T) 


def gwggrad(constC,hC1,hC2,T):
    
    """ Return the gradient for Gromov-Wasserstein
    The gradient is computed as described in Proposition 2 in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    grad : ndarray, shape (ns, nt)
           Gromov Wasserstein gradient
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
          
    return 2*tensor_product(constC,hC1,hC2,T) 

def gw_lp(C1,C2,p,q,loss_fun='square_loss',alpha=1,amijo=True,**kwargs): 

    """
    Returns the gromov-wasserstein transport between (C1,p) and (C2,q)
    The function solves the following optimization problem:
    .. math::
        \GW_Dist = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        p  : distribution in the source space
        q  : distribution in the target space
        L  : loss function to account for the misfit between the similarity matrices
        H  : entropy
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string
        loss function used for the solver
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    amijo : bool, optional
        If True the step of the line-search is found via an amijo research. Else closed form is used.
        If there is convergence issues use False.
    **kwargs : dict
        parameters can be directly pased to the ot.optim.cg solver
    Returns
    -------
    T : ndarray, shape (ns, nt)
        coupling between the two spaces that minimizes :
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
    log : dict
        convergence information and loss
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    .. [2] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.
    """

    constC,hC1,hC2=init_matrix(C1,C2,p,q,loss_fun)
    M=np.zeros((C1.shape[0],C2.shape[0]))
    
    G0=p[:,None]*q[None,:]
    
    def f(G):
        return gwloss(constC,hC1,hC2,G)
    def df(G):
        return gwggrad(constC,hC1,hC2,G)
 
    return optim.cg(p,q,M,alpha,f,df,G0,amijo=amijo,constC=constC,C1=C1,C2=C2,**kwargs)
    
def fgw_lp(M,C1,C2,p,q,loss_fun='square_loss',alpha=1,amijo=True,G0=None,**kwargs): 
    """
    Computes the FGW distance between two graphs see [3]
    .. math::
        \gamma = arg\min_\gamma (1-\alpha)*<\gamma,M>_F + alpha* \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
        s.t. \gamma 1 = p
             \gamma^T 1= q
             \gamma\geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)
    The algorithm used for solving the problem is conditional gradient as discussed in  [1]_
    Parameters
    ----------
    M  : ndarray, shape (ns, nt)
         Metric cost matrix between features across domains
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix respresentative of the structure in the source space
    C2 : ndarray, shape (nt, nt)
         Metric cost matrix espresentative of the structure in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string,optionnal
        loss function used for the solver 
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    amijo : bool, optional
        If True the steps of the line-search is found via an amijo research. Else closed form is used.
        If there is convergence issues use False.
    **kwargs : dict
        parameters can be directly pased to the ot.optim.cg solver
    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    constC,hC1,hC2=init_matrix(C1,C2,p,q,loss_fun)
    
    if G0 is None:
        G0=p[:,None]*q[None,:]
    
    def f(G):
        return gwloss(constC,hC1,hC2,G)
    def df(G):
        return gwggrad(constC,hC1,hC2,G)
 
    return optim.cg(p,q,M,alpha,f,df,G0,amijo=amijo,C1=C1,C2=C2,constC=constC,**kwargs)



def update_square_loss(p, lambdas, T, Cs):
    """
    Updates C according to the L2 Loss kernel with the S Ts couplings
    calculated at each iteration
    Parameters
    ----------
    p  : ndarray, shape (N,)
         masses in the targeted barycenter
    lambdas : list of float
              list of the S spaces' weights
    T : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Cs : list of S ndarray, shape(ns,ns)
         Metric cost matrices
    Returns
    ----------
    C : ndarray, shape (nt,nt)
        updated C matrix
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s]) for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_cross_feature_matrix(X,Y):

    """
    Updates M the distance matrix between the features
    calculated at each iteration    
    ----------
    X : ndarray, shape (N,d)
        First features matrix, N: number of samples, d: dimension of the features
    Y : ndarray, shape (M,d)
        Second features matrix, N: number of samples, d: dimension of the features
    Returns
    ----------
    M : ndarray, shape (N,M)
    
    """

    return ot.dist(reshaper(np.array(X)),reshaper(np.array(Y)))

def update_Ms(X,Ys):

    l=[np.asarray(update_cross_feature_matrix(X,Ys[s]), dtype=np.float64) for s in range(len(Ys))]

    return l


def random_gamma_init(p,q, **kwargs):
    rvs=stats.beta(1e-1,1e-1).rvs
    S=random(len(p), len(q), density=1, data_rvs=rvs)
    return sinkhorn_scaling(p,q,S.A, **kwargs)


def update_feature_matrix(lambdas,Ys,Ts,p):
    
    """
    Updates the feature with respect to the S Ts couplings. See "Solving the barycenter problem with Block Coordinate Descent (BCD)" in [3]
    calculated at each iteration
    Parameters
    ----------
    p  : ndarray, shape (N,)
         masses in the targeted barycenter
    lambdas : list of float
              list of the S spaces' weights
    Ts : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Ys : list of S ndarray, shape(d,ns)
         The features
    Returns
    ----------
    X : ndarray, shape (d,N)
    
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    p=np.diag(np.array(1/p).reshape(-1,))

    tmpsum = sum([lambdas[s] * np.dot(Ys[s],Ts[s].T).dot(p) for s in range(len(Ts))])

    return tmpsum

def fgw_barycenters(N,Ys,Cs,ps,lambdas,alpha,fixed_structure=False,fixed_features=False,p=None,loss_fun='square_loss',
                    max_iter=100, tol=1e-9,verbose=False,log=True,init_C=None,init_X=None):
 
    """
    Compute the fgw barycenter as presented eq (5) in [3].
    ----------
    N : integer 
        Desired number of samples of the target barycenter
    Ys: list of ndarray, each element has shape (ns,d)
        Features of all samples
    Cs : list of ndarray, each element has shape (ns,ns)
         Structure matrices of all samples
    ps : list of ndarray, each element has shape (ns,)
        masses of all samples
    lambdas : list of float
              list of the S spaces' weights
    alpha : float
            Alpha parameter for the fgw distance
    fixed_structure :  bool
                       Wether to fix the structure of the barycenter during the updates
    fixed_features :  bool
                       Wether to fix the feature of the barycenter during the updates
    init_C :  ndarray, shape (N,N), optional 
              initialization for the barycenters' structure matrix. If not set random init
    init_X :  ndarray, shape (N,d), optional 
              initialization for the barycenters' features. If not set random init
    Returns
    ----------
    X : ndarray, shape (N,d)
        Barycenters' features
    C : ndarray, shape (N,N)
        Barycenters' structure matrix
    log_:
        T : list of (N,ns) transport matrices
        Ms : all distance matrices between the feature of the barycenter and the other features dist(X,Ys) shape (N,ns)
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    np.random.seed(2)
    S = len(Cs)
    d= reshaper(Ys[0]).shape[1] #dimension on the node features
    if p is None:
        p=np.ones(N)/N

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    Ys = [np.asarray(Ys[s], dtype=np.float64) for s in range(S)]

    lambdas = np.asarray(lambdas, dtype=np.float64)

    if fixed_structure:
        if init_C is None:
            C=Cs[0]
        else:
            C=init_C
    else:
        if init_C is None:
            xalea = np.random.randn(N, 2)
            C = dist(xalea, xalea)
            C /= C.max()
        else:
            C = init_C

    if fixed_features:
        if init_X is None:
            X=Ys[0]
        else :
            X= init_X
    else:
        if init_X is None: 
            X=np.zeros((N,d))
        else:
            X = init_X

    #T=[np.outer(p,q) for q in ps]
    T=[random_gamma_init(p,q) for q in ps]

    # X is N,d
    # Ys is ns,d
    Ms = update_Ms(X,Ys)
    # Ms is N,ns

    cpt = 0
    err_feature = 1
    err_structure = 1

    if log:
        log_={}
        log_['err_feature']=[]
        log_['err_structure']=[]
        log_['Ts_iter']=[]

    while((err_feature > tol or err_structure > tol) and cpt < max_iter):
        Cprev = C
        Xprev = X

        if not fixed_features:
            Ys_temp=[reshaper(y).T for y in Ys] 
            X=update_feature_matrix(lambdas,Ys_temp,T,p)

        # X must be N,d
        # Ys must be ns,d
        Ms=update_Ms(X.T,Ys)

        if not fixed_structure:
            if loss_fun == 'square_loss':
                # T must be ns,N
                # Cs must be ns,ns
                # p must be N,1
                T_temp=[t.T for t in T]
                C = update_square_loss(p, lambdas, T_temp, Cs)

        # Ys must be d,ns
        # Ts must be N,ns
        # p must be N,1
        # Ms is N,ns
        # C is N,N 
        # Cs is ns,ns
        # p is N,1
        # ps is ns,1

        T = [fgw_lp((1-alpha)*Ms[s],C,Cs[s],p,ps[s],loss_fun,alpha,numItermax=max_iter, stopThr=1e-5, verbose=verbose) for s in range(S)]

            # T is N,ns

        log_['Ts_iter'].append(T)
        err_feature = np.linalg.norm(X - Xprev.reshape(d,N))
        err_structure = np.linalg.norm(C - Cprev)

        if log:
            log_['err_feature'].append(err_feature)
            log_['err_structure'].append(err_structure)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err_structure))
            print('{:5d}|{:8e}|'.format(cpt, err_feature))

        cpt += 1
    log_['T']=T # ce sont les matrices du barycentre de la target vers les Ys
    log_['p']=p
    log_['Ms']=Ms #Ms sont de tailles N,ns

    return X.T,C,log_