# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 19:18:51 2026

@author: Elena Antonelli



Semi-supervised Spectral Clustering with soft pairwise constraints.

Multi-class embedding is obtained by solving a generalized eigenproblem:
    L v = λ (Q - beta I) v
and clustering the K-dimensional spectral embedding with K-means.

"""

# Import the libraries
import numpy as np
import scipy.linalg as la

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def _as_pairs_with_weights(pairs, default_w = 1.0):
    """
    This function converts a set of pairwise constraints into a
    standard format.

    :arg(pairs, default_w):
        pairs = collection of pairwise constraints or None
        default_w = weight used when pairs are unweighted

    :return:
        ij = np.ndarray containing the index pairs
        w  = np.ndarray containing one weight per pair
    """
    ij = np.zeros((0, 2), dtype = int)
    w = np.zeros((0,), dtype = float)
    
    if pairs is None:
        return ij, w

    arr = np.asarray(pairs)
    if arr.size == 0:
        return ij, w

    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            "pairs must be shaped (m,2) or (m,3) with (i,j[,w])."
            ) # m = number of constraints
                        
    ij = arr[:, :2].astype(int, copy = False)
    if arr.shape[1] == 3:
        w = arr[:, 2].astype(float, copy = False)
    else:
        w = np.full((ij.shape[0],), float(default_w), dtype = float)

    return ij, w


def _build_W(X, graph_type = "rbf", gamma = None, n_neighbors = 10,
             include_self = False):
    """
    This function builds the matrix W associated with the data graph.

    :arg(X, graph_type, gamma, n_neighbors, include_self):
        X = collection of data points
        graph_type = type of graph ("rbf" or "knn")
        gamma = scale parameter of the RBF kernel 
                (automatically estimated if None)
        n_neighbors = number of neighbors used in the kNN graph
        include_self = whether self-connections are included

    :return: W, the symmetric matrix encoding pairwise similarities
    """
    X = np.asarray(X)
    P = X.shape[0]

    if graph_type == "rbf":
        d = pairwise_distances(X, metric = "euclidean", n_jobs = None)
        d2 = d * d
        if gamma is None:
        # gamma controls how fast the weight falls with distance
            med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
            gamma = 1.0 / med
        W = np.exp(-gamma * d2)

        if not include_self:
            np.fill_diagonal(W, 0.0) # W_ii = 0

        W = 0.5 * (W + W.T)
        return W.astype(float)

    if graph_type == "knn": # (sparse graph)
        nn = NearestNeighbors(
            n_neighbors = n_neighbors, 
            metric = "cosine"
            )  
        nn.fit(X)
        dists, idx = nn.kneighbors(X)

        # gamma: if None, use median of distances^2 to neighbors
        d2 = dists**2
        if gamma is None:
            med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
            gamma = 1.0 / med

        W = np.zeros((P, P), dtype = float)
        for i in range(P):
            for jj, dist2 in zip(idx[i], d2[i]):
                if (not include_self) and (jj == i):
                    continue
                w = np.exp(-gamma * dist2)
                W[i, jj] = max(W[i, jj], w)

        W = 0.5 * (W + W.T)
        return W
        
    raise ValueError("graph type must be one of: {'rbf','knn'}")


def _laplacian(W, normalized):
    """
    This function builds the graph Laplacian matrix 
    starting from a similarity matrix W.

    :arg(W, normalized):
        W = input matrix encoding relationships between data points
        normalized = if True use the normalized Laplacian, 
                     otherwise use the unnormalized one

    :return:
        L = Laplacian matrix
        d = degree vector
    """
    W = np.asarray(W, dtype = float)
    d = W.sum(axis=1) # d = vector of node degrees (D_ii)

    if normalized:
        with np.errstate(divide = "ignore"): 
        # to avoid warnings appearing in console
            D_inv_sqrt = np.where(d > 1e-12, 1.0 / np.sqrt(d), 0.0)
        S = (D_inv_sqrt[:, None] * W) * D_inv_sqrt[None, :]
        L = np.eye(W.shape[0]) - S
        return L, d
    else:
        L = np.diag(d) - W
        return L, d


def _build_Q(P, must_link = None, cannot_link = None, 
             must_weight = 1.0, cannot_weight = 1.0):
    """
    This function builds the constraint matrix Q
    encoding pairwise supervision.

    :arg(P, must_link, cannot_link, must_weight, cannot_weight):
        P = number of data points (size of the matrix Q is P x P)
        must_link = collection of must-link constraints 
                    (pairs, optionally with weights)
        cannot_link = collection of cannot-link constraints 
                     (pairs, optionally with weights)
        must_weight = default weight assigned to must-link constraints 
                      when unweighted
        cannot_weight = default weight assigned to cannot-link 
                        constraints when unweighted

    :return: Q (dense, symmetric, zero diagonal)
    """
    ml_ij, ml_w = _as_pairs_with_weights(must_link, 
                                         default_w = must_weight)
    cl_ij, cl_w = _as_pairs_with_weights(cannot_link, 
                                         default_w = cannot_weight)

    Q = np.zeros((P, P), dtype=float)

    for (i, j), w in zip(ml_ij, ml_w):
        Q[i, j] += w
        Q[j, i] += w

    for (i, j), w in zip(cl_ij, cl_w):
        Q[i, j] -= w
        Q[j, i] -= w

    np.fill_diagonal(Q, 0.0)
    return Q


class SpectralSSC:
    def  __init__(self, n_clusters, normalized = True, 
                  graph_type = "rbf", gamma = None, 
                  n_neighbors = 10, delta = 0.0, beta_search = True, 
                  beta_max_eig_margin = 1e-6, beta_search_iters = 30, 
                  random_state = 0, kmeans_n_init = 20):
        # # Graph construction parameters
        self.n_clusters = int(n_clusters)   # number of clusters K
        self.normalized = bool(normalized)  # for normalized Laplacian
        self.graph_type = graph_type        # "rbf", "knn"
        self.gamma = gamma                  # RBF scale 
        self.n_neighbors = int(n_neighbors) # neighbors for knn

        # Constraint-related parameters
        self.delta = float(delta)  # constraint lower bound
        self.beta_search = bool(beta_search) # enable beta search
        self.beta_max_eig_margin = float(beta_max_eig_margin) 
        # beta_max_eig_margin = beta safety margin
        self.beta_search_iters = int(beta_search_iters) 

        # Optimization / reproducibility
        self.random_state = random_state    # random seed 
        self.kmeans_n_init = int(kmeans_n_init) # k-means restarts

        # Learned attributes
        self.labels_ = None     # cluster labels 
        self.embedding_ = None  # spectral embedding
        self.beta_ = None       # selected beta

    def fit(self, X, constraints = None, must_link = None,
            cannot_link = None, must_weight = 1.0, 
            cannot_weight = 1.0):       
        """
        This function fits the semi-supervised spectral clustering 
        model to the input data X and stores the resulting clustering.
    
        :arg(X, constraints, must_link, cannot_link, must_weight,
             cannot_weight):
            X = data matrix 
            constraints = dict of must-link / cannot-link constraints
            must_link = collection of must-link index pairs (or None)
            cannot_link = collection of cannot-link index pairs (or None)
            must_weight = default weight assigned to must-link 
                          constraints when unweighted
            cannot_weight = default weight assigned to cannot-link 
                            constraints when unweighted
        """
        X = np.asarray(X)
        P = X.shape[0]
        K = int(self.n_clusters)
        if K < 2:
            raise ValueError("n_clusters must be >= 2")

        if constraints is not None:
            must_link = constraints.get("must_link", must_link) 
            cannot_link = constraints.get("cannot_link", cannot_link)

        W = _build_W(X, graph_type = self.graph_type, gamma = self.gamma, 
                     n_neighbors = self.n_neighbors)
        L, d = _laplacian(W, normalized = self.normalized)
        
        Q = _build_Q(
            P,
            must_link = must_link,
            cannot_link = cannot_link,
            must_weight = must_weight,
            cannot_weight = cannot_weight,
        )
       
        no_constraints = (np.count_nonzero(Q) == 0)
        if no_constraints:
            V = self._unsupervised_embedding(L, d, K)
            labels = self._cluster_rows(V)
            self.embedding_ = V
            self.labels_ = labels
            self.beta_ = None
            return self

        V, beta = self._constrained_embedding(L, d, Q, K, 
                                              delta = self.delta)

        labels = self._cluster_rows(V)
        self.embedding_ = V
        self.labels_ = labels
        self.beta_ = beta
        return self

    def fit_predict(self, X, **kwargs):       
        """
        This method fits the model to the input data X and
        returns the resulting cluster labels.
    
        :arg(X, **kwargs):
            X = data matrix 
            **kwargs = additional arguments forwarded to "fit"
                          
        :return: array of cluster labels assigned to each data point
        """
        self.fit(X, **kwargs)
        return self.labels_

    def _cluster_rows(self, V):
        """
        This method assigns cluster labels by applying K-means 
        to the spectral embedding.
   
        :arg(V):
            V = spectral embedding matrix (one row per data point)
    
        :return: array of cluster assignments for each data point 
                 (labels)
        """        
        # row-normalize embedding for K-means stability 
        if self.normalized: # with normalized Laplacian
            norms = np.linalg.norm(V, axis = 1, keepdims = True) 
            
            # avoid division by zero
            norms = np.maximum(norms, 1e-12) 
            Z = V / norms
        else:
            Z = V
        
        km = KMeans(
            n_clusters = self.n_clusters, 
            n_init = self.kmeans_n_init, 
            random_state = self.random_state
            )
        self.labels_ = km.fit_predict(Z)       
        return self.labels_

    def _unsupervised_embedding(self, L, d, K):      
        """
        Method that computes the unsupervised spectral embedding.
    
        :arg(L, d, K):
            L = graph Laplacian matrix
            d = degree vector 
            K = target embedding dimension 
                (equals the number of clusters)
    
        :return: V = embedding matrix (one row per data point)
        """
        # Dense spectral embedding 
        evals, evecs = np.linalg.eigh(np.asarray(L, dtype = float))
        V = evecs[:, :K + 1]

        # project out the trivial (constant) direction
        if self.normalized and d is not None:
            t = np.sqrt(np.maximum(d, 0.0))
        else:
            t = np.ones(V.shape[0], dtype = float)
       
        nt = np.linalg.norm(t)
        if nt >= 1e-12:
            t = t / nt
    
            # project V onto the subspace orthogonal to t
            V = V - np.outer(t, t @ V) 

        V, _ = np.linalg.qr(V) # V with orthonormal columns
        return V[:, :K]

    def _constraint_score(self, V, Q):       
        """
        This method measures how well an embedding V satisfies 
        the pairwise constraints encoded in Q.
    
        :arg(V, Q):
            V = embedding matrix (P x K), 
                whose columns are embedding directions
            Q = constraint matrix (P x P)
        
        :return: scalar value measuring the average constraint 
                 satisfaction in V
        """
        # average quadratic satisfaction: trace(V^T Q V) / K
        K = V.shape[1]
        M = Q @ V
        return float(np.trace(V.T @ M) / max(K, 1))

    def _constrained_embedding(self, L, d, Q, K, delta):         
        """
        This function computes a constraint-aware spectral embedding 
        and (optionally) selects beta.
    
        :arg(L, d, Q, K, delta):
            L = graph Laplacian matrix
            d = degree vector 
            Q = constraint matrix encoding must-link (positive) 
                and cannot-link (negative)
            K = target embedding dimension
            delta = minimum required constraint satisfaction score
    
        :return:
            V = embedding matrix 
            beta = selected beta value 
        """
        # generalized eigenproblem
        #  L v = lambda (Q - beta I) v
       
        # beta is chosen so that score(V) >= delta
        # score(V) = (1/K) * trace(V^T Q V).
        
        P = L.shape[0]
        I = np.eye(P)
      
        # Minimum eigenvalue of Q 
        min_eig_Q = float(np.linalg.eigvalsh(np.asarray(
            Q, dtype = float)).min())
        
        # Choosing the initial range for beta
        beta_hi = min_eig_Q - abs(self.beta_max_eig_margin)
        # beta_hi to avoid the point where Q − beta I becomes singular
        beta_lo = beta_hi - 100.0

        def embedding_for_beta(beta):
            B = Q - beta * I
            # solve L v = lambda B v 
            vals, vecs = la.eigh(np.asarray(
                L, dtype = float), np.asarray(B, dtype = float))            
            V = vecs[:, :K + 1]
            
            # project out the trivial (constant) direction
            if self.normalized and d is not None:
                t = np.sqrt(np.maximum(d, 0.0))
            else:
                t = np.ones(V.shape[0], dtype = float)
            nt = np.linalg.norm(t)
            if nt >= 1e-12:
                t = t / nt
                V = V - np.outer(t, t @ V)
    
            V, _ = np.linalg.qr(V)
            return V[:, :K]

        # initial try
        V0 = embedding_for_beta(beta_hi)

        if not self.beta_search:
        # if the user does not want to search for beta, stop here
            return V0, float(beta_hi)

        s0 = self._constraint_score(V0, Q)
        if s0 >= delta:
            return V0, float(beta_hi)

        # try to find a feasible lower beta
        Vlo, slo = None, -np.inf
        scale = max(1.0, abs(min_eig_Q))
        beta_lo = beta_hi - 1.0 * scale
        
        for _ in range(6):
            Vlo = embedding_for_beta(beta_lo)
            slo = self._constraint_score(Vlo, Q)
            if slo >= delta:
                break
            beta_lo -= 2.0 * scale
            
        # Keep track of the best embedding found so far
        bestV, best_beta, bestS = V0, float(beta_hi), float(s0)
        if Vlo is not None and slo > bestS:
            bestV, best_beta, bestS = Vlo, float(beta_lo), float(slo)
        if bestS >= delta:
            return bestV, best_beta

        # 1D search on beta (bisection)
        lo, hi = float(beta_lo), float(beta_hi)
        s_hi = float(s0)

        for _ in range(int(self.beta_search_iters)):
            mid = 0.5 * (lo + hi)
            Vm = embedding_for_beta(mid)
            sm = float(self._constraint_score(Vm,Q))

            if sm > bestS:
                bestV, best_beta, bestS = Vm, float(mid), sm

            if sm >= s_hi:
                lo = mid
                s_hi = sm
            else:
                hi = mid

        return bestV, best_beta
