"Graph clustering using FGW"

from FGW import fgw_barycenters, fgw_lp
import numpy
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist
import sys
import random
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np 
from graph import Graph

def build_comunity_graph(N=30,Nc=3,sigma=0.3,pw=0.8,pb=0.2):
    
    c=(Nc*np.arange(N)/N).astype(int)
    c2=(2*Nc*np.arange(N)/N).astype(int)
    v=c+1*np.mod(c2,2)+sigma*np.random.randn(N);
    g=Graph()
    g.add_nodes(list(range(N)))
    for i in range(N):
         g.add_one_attribute(i,v[i])
         for j in range(i+1,N):
             r=np.random.rand()
             if (c[i]==c[j]) or ((c[i]==c[j]-1) and r<pb): # or (c[i]==0 and c[j]==Nc)
                 g.add_edge((i,j))
         
    return g,v

def cdist_fgw(X_features, X_structure, Y_features, Y_structure, alpha,metric='sqeuclidean'):
    n_X = len(X_features)
    n_Y = len(Y_features)

    # TMP #
    for i in range(n_X):
        assert numpy.linalg.norm(X_structure[i] - X_structure[i].T) < 1e-5, "Wooops X"
    for j in range(n_Y):
        assert numpy.linalg.norm(Y_structure[j] - Y_structure[j].T) < 1e-5, "Wooops Y"


    dists = numpy.empty((n_X, n_Y))
    for i in range(n_X):
        for j in range(n_Y):
            dist_features_ij = cdist(numpy.array(X_features[i]).reshape((len(X_features[i]), -1)),
                                     numpy.array(Y_features[j]).reshape((len(Y_features[j]), -1)),
                                     metric=metric)
            dist_features_ij *= (1. - alpha)
            transport, log = fgw_lp(dist_features_ij,
                                    X_structure[i],
                                    Y_structure[j],
                                    p=numpy.ones(len(X_features[i]), ) / len(X_features[i]),
                                    q=numpy.ones(len(Y_features[j]), ) / len(Y_features[j]),
                                    loss_fun='square_loss',
                                    alpha=alpha,
                                    verbose=False,
                                    log=True)
            # TODO: test wgw ? (reg) regarder GW_dist
            dists[i, j] = log["loss"][-1]

    return dists

def _check_no_empty_cluster(labels, n_clusters):
    """Check that all clusters have at least one sample assigned.

    Examples
    --------
    >>> labels = numpy.array([1, 1, 2, 0, 2])
    >>> _check_no_empty_cluster(labels, 3)
    >>> _check_no_empty_cluster(labels, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    EmptyClusterError: Cluster assignments lead to at least one empty cluster
    """

    for k in range(n_clusters):
        if numpy.sum(labels == k) == 0:
            raise EmptyClusterError

class EmptyClusterError(Exception):
    def __init__(self, message=""):
        super(EmptyClusterError, self).__init__()
        self.message = message

    def __str__(self):
        if len(self.message) > 0:
            suffix = " (%s)" % self.message
        else:
            suffix = ""
        return "Cluster assignments lead to at least one empty cluster" + suffix

class FusedGromovWassersteinGraphKMeans():
    """K-means clustering with FGW for graph data.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.
    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm stops.
    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.
    max_iter_barycenter : int (default: 100)
        Number of iterations for the barycenter computation process.
    metric_params : dict or None
        Parameter values for the chosen metric.
        Value associated to the `"alpha"` key corresponds to the alpha parameter in Fused-Gromov-Wasserstein.
        Default value is 0.5.
        Value associated to the `"centroid_sz"` key defines the size (in number of timestamps) of the obtained centroids.
        Default value is `None` which means the size of the first time series in the dataset will be used.
        Value associated to the `"fixed_structure"` key defines whether to use a fixed (regular) structure for the.
        barycenters or not. Default value is `False`.
        Value associated to the `"line_search_method"` key defines the method to be used for line search during
        optimization. Default value is `"amijo"`.
    verbose : {0, 1, 2} (default: 1)
        Verbose level: 0 means no message, 1 means messages only from the clustering part, 2 means messages from both
        clustering and FGW

    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point.
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """

    def __init__(self,N=None, n_clusters=3, max_iter=50, tol=1e-4, n_init=1, max_iter_barycenter=100,
                 metric_params=None, verbose=1, random_state=None,max_attempts=10,b=0.9,a=0.5):

        if metric_params is None:
            metric_params = {}
        self.alpha_fgw = metric_params.get("alpha", 0.5)
        self.fixed_structure = metric_params.get("fixed_structure", False)
        self.fixed_feature=metric_params.get("fixed_feature", False)
        self.bar_structure=metric_params.get("bar_structure", None)
        self.bar_feature=metric_params.get("bar_feature", None)
        self.line_search_method = metric_params.get("line_search_method", "amijo")
        self.max_attempts=max_attempts
        self.max_iter=max_iter

        self.n_clusters=n_clusters

        self.max_iter_barycenter=max_iter_barycenter
        self.tol=tol
        self.n_init=n_init
        self.verbose=verbose
        self.random_state=random_state

        self._cluster_centers_features = None
        self._cluster_centers_structure = None
        self.labels_=None
        self.inertia_ = numpy.inf
        self.all_cluster_centers_features={}
        self.all_cluster_centers_structure={}
        self.N=N

        self.a=a
        self.b=b

    def compute_inertia(self,distances,assignments):
        n=distances.shape[0]
        return numpy.sum(distances[numpy.arange(n), assignments]) / n

    def _assign_fgw(self, X, structural_information, update_class_attributes=True):

        #print('_assign_fgw')

        dists = cdist_fgw(X_features=[x.values() for x in X],
                          X_structure=structural_information,
                          Y_features=self._cluster_centers_features,
                          Y_structure=self._cluster_centers_structure,
                          alpha=self.alpha_fgw)
        matched_labels = dists.argmin(axis=1)

        if update_class_attributes:
            #print("update_class_attributes")
            self.labels_ = matched_labels
            _check_no_empty_cluster(self.labels_, self.n_clusters)
            inertia_dists = dists
            self.inertia_ = self.compute_inertia(inertia_dists, self.labels_)
        return matched_labels

    def _update_centroids(self,X,it):

        for k in range(self.n_clusters):
            cluster_data = [x.values() for x in X[self.labels_ == k]]
            structural_information = [x.C for x in X[self.labels_ == k]]
            if self.N is None:
                centroid_mean_size=int(numpy.mean([len(x.nodes()) for x in X[self.labels_ == k]])) #size of the centroid in the cluster is the mean size of the nodes of the graphs in the cluster
            else:    
                centroid_mean_size=self.N

            internal_weights = [numpy.ones(len(x.nodes())) / len(x.nodes()) for x in X[self.labels_ == k]]
            graph_weights = numpy.ones(len(internal_weights)) / len(internal_weights)
                        
            features, structure, _ = fgw_barycenters(N=centroid_mean_size,
                                                     Ys=cluster_data,
                                                     Cs=structural_information,
                                                     ps=internal_weights,
                                                     lambdas=graph_weights,
                                                     alpha=self.alpha_fgw,
                                                     max_iter=self.max_iter_barycenter,
                                                     fixed_structure=False,
                                                     fixed_features=False,
                                                     init_X=self._cluster_centers_features[k],
                                                     init_C=self._cluster_centers_structure[k],
                                                     verbose=(self.verbose == 2))
            self._cluster_centers_features[k] = features
            self._cluster_centers_structure[k] = structure

            self.all_cluster_centers_features[(it,k)]=features
            self.all_cluster_centers_structure[(it,k)]=structure

    def _fit_one_init_fgw(self, X, structural_information, rs,y=None):

        breaked=False
        
        #indices = rs.randint(low=0, high=len(X), size=self.n_clusters)
        #self._cluster_centers_features = [X[i].values() for i in indices]
        #self._cluster_centers_structure = [X[i].C for i in indices]

        self._cluster_centers_features=[]
        self._cluster_centers_structure=[]
        for k in range(self.n_clusters):
            if self.N is None:
                N=np.random.randint(15,20)
            else:
                N=self.N
            #Nc=np.random.randint(3,5)
            Nc=3
            g,v=build_comunity_graph(N,Nc,sigma=0.1,pw=0.7,pb=0.6)
            #self._cluster_centers_features.append((self.b-self.a)*np.random.rand(N)+self.a)
            #self._cluster_centers_structure.append(nx.adjacency_matrix(nx.fast_gnp_random_graph(N,1)).toarray())
            self._cluster_centers_features.append(v)
            C=g.distance_matrix(method='shortest_path',force_recompute=True)
            self._cluster_centers_structure.append(C)




        for k in range(self.n_clusters):
            self.all_cluster_centers_features[(0,k)]=self._cluster_centers_features[k]
            self.all_cluster_centers_structure[(0,k)]=self._cluster_centers_structure[k]

        old_inertia = numpy.inf

        for it in range(self.max_iter):
            self._assign_fgw(X, structural_information=structural_information)
            if self.verbose > 0:
                sys.stdout.write(" Inertia : %.3f  " % self.inertia_)
                if y is not None:
                    sys.stdout.write(" & MI : %.3f " % adjusted_mutual_info_score(y,self.labels_))
                sys.stdout.write(" -->  ")
                sys.stdout.flush()
            self._update_centroids(X,it+1)

            if numpy.abs(old_inertia - self.inertia_) < self.tol:
                self.attempted_iter=it+1
                breaked=True
                break
            old_inertia = self.inertia_
        if not breaked:
            self.attempted_iter=self.max_iter+1
        if self.verbose > 0:
            sys.stdout.write("\n")

        return self

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        """
        structural_information = [x.C for x in X]


        rs = check_random_state(self.random_state)

        best_correct_centroids = None
        min_inertia = numpy.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < self.max_attempts:
            try:
                if self.verbose > 0 and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init_fgw(X, structural_information, rs,y=y)

                if self.inertia_ < min_inertia:
                    best_correct_centroids = (self._cluster_centers_features.copy(),
                                              self._cluster_centers_structure.copy())
                    min_inertia = self.inertia_
                n_successful += 1
                n_successful += 1

            except EmptyClusterError:
                if self.verbose > 0:
                    print("Resumed because of empty cluster")
        self._post_fit_fgw(X, structural_information, best_correct_centroids, min_inertia)
        

        return self

    def predict(self, X_tuple):
        """Compute assignment for data in X_tuple.

        Parameters
        ----------
        X_tuple : pair of (feature matrix, list_of_structure_matrices)

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X_, structural_information = X_tuple
        return self._assign_fgw(X_, structural_information=structural_information, update_class_attributes=False)

    def _post_fit_fgw(self, X_fitted, structural_information_fitted, centroids, inertia):
        if numpy.isfinite(inertia) and (centroids is not None):
            self._cluster_centers_features, self._cluster_centers_structure = centroids
            self._assign_fgw(X_fitted, structural_information_fitted)
            #self._compute_cluster_centers(X_fitted=X_fitted)
            self.X_fit_ = X_fitted
            self.inertia_ = inertia
        else:
            self.X_fit_ = None


class FusedGromovWassersteinGraphKMedois():

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-6, n_init=1,
                 metric_params=None, verbose=1, random_state=None):

        if metric_params is None:
            metric_params = {}
        self.alpha_fgw = metric_params.get("alpha", 0.5)
        self.metric=metric_params.get("feature_metric", "euclidean")

        self.max_iter=max_iter

        self.n_clusters=n_clusters

        self.tol=tol
        self.n_init=n_init
        self.verbose=verbose
        self.random_state=random_state

        self._cluster_centers_features = None
        self._cluster_centers_structure = None
        self.labels_=None
        self.inertia_ = numpy.inf
        
        self.distances=None
        
    def fit(self,X,y=None):
        
        X=numpy.array(X)
        
        if self.distances is None :     
            dists= numpy.zeros((X.shape[0], X.shape[0]))
            H=numpy.zeros((X.shape[0], X.shape[0]))
            
            for i, x1 in enumerate(X):
                for j,x2 in enumerate(X):
                    if j>=i: 
                        dist_features_ij = cdist(numpy.array(X[i].values()).reshape((len(X[i].values()), -1)),
                                                 numpy.array(X[j].values()).reshape((len(X[j].values()), -1)),
                                                 metric=self.metric)
                        dist_features_ij *= (1. - self.alpha_fgw)
                        transport, log = fgw_lp(dist_features_ij,
                                                X[i].C,
                                                X[j].C,
                                                p=numpy.ones(len(X[i].values()), ) / len(X[i].values()),
                                                q=numpy.ones(len(X[j].values()), ) / len(X[j].values()),
                                                loss_fun='square_loss',
                                                alpha=self.alpha_fgw,
                                                verbose=False,
                                                log=True)
                        dists[i, j] = log["loss"][-1]
                        if self.verbose >2:
                            print('Done one distance')
                if self.verbose >1:
                    print('"""""""')
                    print('Done '+str(100*i/X.shape[0])+' %')
                    print('"""""""')

            numpy.fill_diagonal(H,numpy.diagonal(dists))
            dists=dists+dists.T-H
            
            self.distances=dists
            
            print('All distances are computed')

    def predict(self):
    
        m = self.distances.shape[0] # number of points
    
        # Pick k random medoids.
        curr_medoids = numpy.array([-1]*self.n_clusters)
        while not len(numpy.unique(curr_medoids)) == self.n_clusters:
            curr_medoids = numpy.array([random.randint(0, m - 1) for _ in range(self.n_clusters)])
        old_medoids = numpy.array([-1]*self.n_clusters) # Doesn't matter what we initialize these to.
        new_medoids = numpy.array([-1]*self.n_clusters)
       
        # Until the medoids stop updating, do the following:
        while not ((old_medoids == curr_medoids).all()):
            # Assign each point to cluster with closest medoid.
            clusters = self.assign_points_to_clusters(curr_medoids)
    
            # Update cluster medoids to be lowest cost point. 
            for curr_medoid in curr_medoids:
                cluster = numpy.where(clusters == curr_medoid)[0]
                new_medoids[curr_medoids == curr_medoid] = self.compute_new_medoid(cluster)
    
            old_medoids[:] = curr_medoids[:]
            curr_medoids[:] = new_medoids[:]
    
        return clusters, curr_medoids
    
    def assign_points_to_clusters(self,medoids):
        distances_to_medoids = self.distances[:,medoids]
        clusters = medoids[numpy.argmin(distances_to_medoids, axis=1)]
        clusters[medoids] = medoids
        return clusters
    
    def compute_new_medoid(self,cluster):
        mask = numpy.ones(self.distances.shape)
        mask[numpy.ix_(cluster,cluster)] = 0.
        cluster_distances = numpy.ma.masked_array(data=self.distances, mask=mask, fill_value=10e9)
        costs = cluster_distances.sum(axis=1)
        return costs.argmin(axis=0, fill_value=10e9)





