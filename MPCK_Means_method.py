# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 18:13:49 2026

@author: Elena Antonelli
"""

# Import the libraries
import numpy as np

class MPCKMeans:
    def __init__(self, K, w = 1.0, max_iter = 150, random_state = None,
                 eps0 = 1e-9, tol = 1e-6, reg_max_tries = 6,
                 reg_growth = 10.0):
        self.K = K               # number of clusters
        self.w = w               # weight of pairwise constraints
        self.max_iter = max_iter # maximum number of iters
        self.eps0 = eps0         # initial regularization parameter 
        self.tol = tol           # convergence tolerance
        self.reg_max_tries = reg_max_tries 
        # reg_max_tries = max attempts if matrix not SPD 
        self.reg_growth = reg_growth # multiplicative factor 
        self.rng = np.random.default_rng(random_state) 
        # RNG for reproducibility
           
    def fit(self, X, must_link = (), cannot_link = ()):      
        """
        This method fits the MPCK-Means model to the input dataset X  
        under soft pairwise Must-Link and Cannot-Link constraints.
    
        :arg(X, must_link, cannot_link):
            X = data matrix of shape (P, N)
            must_link = index pairs (i, j) that should be in the 
                        same cluster
            cannot_link = index pairs (i, j) that should be in 
                          different clusters
         
        :return:
            labels = array of cluster assignments for each data point 
            mu = matrix of final cluster centroids 
            M = learned cluster-specific inverse metric matrices A_k^{-1}
            neighborhoods = list of connected components induced 
                            by Must-Link constraints
        """
        X = np.asarray(X, dtype = float)
        P, N = X.shape

        # Preprocess constraints: neighborhoods + ML/CL propagation
        ml_graph, cl_graph, neighborhoods = self._preprocess_constraints(
            P, list(must_link), list(cannot_link))

        # Initialize centroids using neighborhoods
        mu = self._initialize_cluster_centers(X, neighborhoods)

        # Initialize precision matrices M_k = A_k^{-1} 
        M = np.repeat(np.eye(N)[None, :, :], self.K, axis = 0) 

        # Initial labels via Euclidean distance
        labels = self._initialize_labels(X, mu)

        for _ in range(self.max_iter):
            # save old centroids for convergence checking
            mu_prev = mu.copy() 

            # Ensure SPD matrices and compute Cholesky factors L_k
            # such that M_k = L_k L_k^T
            # used for Mahalanobis distances and log-determinant terms
            cholesky_factors, logdet_M = self._factorize_metrics(M)

            # Compute farthest intra-cluster pairs (for CL term)
            farthest = self._find_farthest_pairs(
                X, labels, cholesky_factors
                )

            # Batch assignment step 
            new_labels = self._assign_clusters(
                X, labels, mu, cholesky_factors, logdet_M, farthest, 
                ml_graph, cl_graph
            )

            # Repair empty clusters 
            new_labels = self._repair_empty_clusters(
                X, new_labels, mu, cholesky_factors
                )

            # Update centroids
            mu = self._update_centroids(X, new_labels)

            # Update precision matrices 
            M = self._update_metrics(
                X, new_labels, mu, cholesky_factors, farthest,
                ml_graph, cl_graph
            )

            labels = new_labels

            # Convergence check: stop if centroids have converged
            if np.allclose(mu, mu_prev, atol=self.tol): 
                break

        return labels, mu, M, neighborhoods

    def _preprocess_constraints(self, P, must_link, cannot_link):       
        """
        This method preprocesses the pairwise constraints by 
        constructing the Must-Link and Cannot-Link graphs and 
        extracting Must-Link connected components (neighborhoods).
    
        :arg(P, must_link, cannot_link):
            P = number of data points
            must_link = pairs (i, j) that should belong to the 
                        same cluster
            cannot_link = pairs (i, j) that should belong to
                          different clusters
    
        :return:
            ml_graph = dictionary encoding the Must-Link graph
            cl_graph = dictionary encoding the Cannot-Link graph
            neighborhoods = list of Must-Link connected components
        """
        ml_graph = {i: set() for i in range(P)}
        cl_graph = {i: set() for i in range(P)}
       
        # Load base constraints
        for i, j in must_link:
            if i != j:
                ml_graph[i].add(j) 
                ml_graph[j].add(i)

        for i, j in cannot_link:
            if i != j:
                cl_graph[i].add(j)
                cl_graph[j].add(i)

        # ML connected components = neighborhoods
        # track visited nodes during DFS  
        visited = np.zeros(P, dtype=bool)       
        neighborhoods = []

        for node in range(P):
            if not visited[node] and len(ml_graph[node]) > 0:
                comp = self._dfs(node, ml_graph, visited)
                
                # transitive closure of ML constraints
                for u in comp:
                    for v in comp:
                        if u != v:
                            ml_graph[u].add(v)
                neighborhoods.append(comp)

        # Propagate CL across neighborhoods
        for i, j in cannot_link:
            for i2 in ml_graph[i]:
                cl_graph[i2].add(j)
                cl_graph[j].add(i2)
            for j2 in ml_graph[j]:
                cl_graph[j2].add(i)
                cl_graph[i].add(j2)

        return ml_graph, cl_graph, neighborhoods

    def _dfs(self, start, G, visited):       
        """
        Method that performs a Depth-First Search (DFS) on the 
        Must-Link graph starting from a given node and 
        returns the corresponding connected component.
    
        :arg(start, G, visited):
            start = index of the starting node
            G = must-link graph; 
                dictionary mapping each point to its neighbors
            visited = boolean array tracking already visited nodes
    
        :return: list of node indices forming the connected component
        """
        # dfs is used to explore nodes connected to each other
   
        stack = [start]  # contains the nodes to explore
        visited[start] = True
        comp = []

        while stack: 
            u = stack.pop() # takes the last inserted node
            comp.append(u)
            for v in G[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        return comp
  
    def _initialize_cluster_centers(self, X, neighborhoods):       
        """
        This function initializes cluster centroids using Must-Link 
        neighborhoods when available.        
        If the number of neighborhoods exceeds K, a weighted  
        farthest-first selection is used. If fewer than K neighborhoods 
        exist, the remaining centroids are generated as perturbations 
        of the global centroid.
    
        :arg(X, neighborhoods):
            X = data matrix of shape (P, N)
            neighborhoods = list of Must-Link connected components
    
        :return: initial cluster centroids 
        """
        # If neighborhoods exist, compute one centroid per neighborhood 
        if neighborhoods:
            neigh_centroids = np.vstack(
                [X[idx].mean(axis = 0) for idx in neighborhoods])
            neigh_sizes = np.array(
                [len(idx) for idx in neighborhoods], dtype = float)
            neigh_weights = neigh_sizes / neigh_sizes.sum()
        else:
            neigh_centroids = np.empty((0, X.shape[1]))
            neigh_weights = np.empty((0,))
    
        # Case lambda > K: weighted farthest-first selection
        if neigh_centroids.shape[0] > self.K:
            chosen = self._weighted_farthest_first(
                neigh_centroids, neigh_weights, self.K
                )
            return neigh_centroids[chosen]
        
        centroids = neigh_centroids
        
        # Case lambda < K: add perturbations of the global centroid   
        if centroids.shape[0] < self.K:
            global_centroid = X.mean(axis = 0)
            missing = self.K - centroids.shape[0]
            noise = self.rng.normal(
                0.0, 0.01, size = (missing, X.shape[1])) 
            extra = global_centroid[None, :] + noise   
            if centroids.size:
                centroids = np.vstack([centroids, extra])  
            else:
                centroids = extra
    
        return centroids

    def _weighted_farthest_first(self, points, weights, k):       
        """
        This method selects k representative points using a 
        weighted farthest-first traversal strategy.

        :arg(points, weights, k):
            points = array of candidate points (centroids)
            weights = non-negative weights associated with each point
            k = number of points to select
    
        :return: list of indices of the selected points
        """
        # Weighted farthest-first traversal:
        # - pick the first point according to weights
        # - then iteratively pick the point maximizing 
        #   (weight * distance_to_nearest_selected)
              
        n = points.shape[0] # number of candidate centroids (lambda)
       
        if k <= 0 or n == 0:
            return []
    
        weights = np.asarray(weights, dtype = float)
        # weights must form a valid probability distribution 
        if weights.sum() <= 0:
            weights = np.ones(n, dtype = float) / n
        else:
            weights = weights / weights.sum()
    
        # first centroid selected by weighted sampling
        first = int(self.rng.choice(n, p = weights))
        chosen = [first]
    
        for _ in range(1, k):
            best_j = None
            best_val = -np.inf
    
            chosen_points = points[chosen]
    
            for j in range(n):
                if j in chosen:
                    continue
    
                # distance to the nearest selected centroid (Euclidean)
                dists = np.sqrt(np.sum(
                    (chosen_points - points[j]) ** 2, axis = 1))
                d_min = float(dists.min())
    
                val = float(weights[j]) * d_min
                if val > best_val:
                    best_val = val
                    best_j = j
    
            chosen.append(int(best_j))
    
        return chosen

    def _initialize_labels(self, X, mu):      
        """
        This method assigns an initial cluster label to each data point
        based on the Euclidean distance to the current centroids.
    
        :arg(X, mu):
            X = data matrix of shape (P, N)
            mu = centroid matrix of shape (K, N)
    
        :return: labels = array of initial cluster assignments 
        """        
        # labels[i] = k means that x^i is assigned to cluster k
        
        labels = np.zeros(X.shape[0], dtype = int)
        for i in range(X.shape[0]):
            # d[k] = || x^i - mu_k ||^2
            d = np.sum((mu - X[i]) ** 2, axis = 1)
            labels[i] = int(np.argmin(d))
                   
        return labels
    
    def _regularize_metric(self, M):       
        """
        Function that regularizes a cluster metric matrix to ensure that 
        it is symmetric positive definite (SPD), and computes its
        Cholesky factor and log-determinant in a numerically stable way.
        
        :arg(M):
            M = metric matrix of shape (N, N)
        
        :return:
            L = Cholesky factor such that M_reg = L L^T
            logdet = log-determinant of the regularized matrix
            M_reg = regularized SPD matrix
        """      
        # Symmetrize
        M = 0.5 * (M + M.T)

        # Trace-based regularization 
        tr = np.trace(M)
        if tr <= 0 or not np.isfinite(tr):
            tr = 1.0

        eps = self.eps0
        I = np.eye(M.shape[0])

        for _ in range(self.reg_max_tries):
        # try increasing regularization until SPD
            M_reg = M + eps * tr * I
            try:
                L = np.linalg.cholesky(M_reg) # Cholesky factor
                #  M = LL^T -> det(M) = det(L)^2 
                # log det(M) = log (prod L_ii)^2 = 2 sum log(L_ii)
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                return L, logdet, M_reg
            except np.linalg.LinAlgError: # if M_reg is not SPD
                eps *= self.reg_growth # epsilon increases

        raise RuntimeError("SPD regularization failed")

    def _factorize_metrics(self, M):       
        """
        This method regularizes and factorizes each cluster inverse
        metric matrix.    
           
        :arg(M):
            M = array of shape (K, N, N) containing A_k^{-1}
    
        :return:
            cholesky_factors = array with the Cholesky factors L_k
            logdet = array containing the log-determinants of M_k
        """
        cholesky_factors = np.zeros_like(M)
        # cholesky_factors[k] = L_k (Cholesky factor of M_k)
        logdet = np.zeros(M.shape[0])

        for k in range(M.shape[0]):
            L, ld, Mk = self._regularize_metric(M[k])
            cholesky_factors[k] = L
            logdet[k] = ld
            M[k] = Mk

        return cholesky_factors, logdet

    def _mahalanobis_sq(self, d, L):      
        """
        This function computes the squared Mahalanobis distance
        d^T A d using the Cholesky factor of M = A^{-1}.
    
        :arg(d, L):
            d = difference vector between a data point and the centroid
            L = Cholesky factor of M_k 
    
        :return: squared Mahalanobis distance 
        """
        z = np.linalg.solve(L, d) # z = L^{-1} d (sys: Lz = d)
        v = np.linalg.solve(L.T, z) # v = M^{−1} d = A d
        return d @ v # d^T A d
     
    def _find_farthest_pairs(self, X, labels, cholesky_factors):      
        """
        Method that identifies, for each cluster, the pair of points
        with maximum Mahalanobis distance under cluster-specific metric.    
        The resulting pairs are used in the Cannot-Link term 
        of the objective function.
    
        :arg(X, labels, cholesky_factors):
            X = data matrix of shape (P, N)
            labels = array of current cluster assignments 
            cholesky_factors = array of Cholesky factors 
    
        :return: farthest = dictionary mapping each cluster k to
                           (i_max, j_max, dist_max), or None if the 
                           cluster contains fewer than two points
        """
        farthest = {}
        for k in range(self.K):
            # find the indices of the points assigned to cluster k
            idx = np.where(labels == k)[0]
            if idx.size < 2:
                farthest[k] = None
                continue

            Lk = cholesky_factors[k]
            best = None
            best_d = -np.inf

            for i in range(len(idx)):
                for j in range(i):
                    d = self._mahalanobis_sq(X[idx[i]] - X[idx[j]], Lk)
                    if d > best_d:
                        best_d = d
                        best = (idx[i], idx[j], d)

            farthest[k] = best
        return farthest

    def _assign_clusters(self, X, labels_prev, mu, cholesky_factors, 
                         logdet, farthest, ml_graph, cl_graph):      
        """
        This method performs the assignment step of MPCK-Means.
        
        :arg(X, labels_prev, mu, cholesky_factors, logdet,
             farthest, ml_graph, cl_graph):
            X = data matrix of shape (P, N)
            labels_prev = cluster assignments from the previous iteration
            mu = current centroid matrix of shape (K, N)
            cholesky_factors = Cholesky factors of M_k = A_k^{-1}
            logdet = log-determinants of M_k
            farthest = dictionary of farthest intra-cluster pairs 
                      (for Cannot-Link term)
            ml_graph = Must-Link graph
            cl_graph = Cannot-Link graph
    
        :return: new_labels = updated cluster assignments 
        """       
        new_labels = np.empty_like(labels_prev)

        for i in range(X.shape[0]):
            costs = []
            for k in range(self.K):
                # Calculate the cost of assigning point x^i to cluster k
                cost = self._assignment_cost(
                    X, i, k, labels_prev, mu, cholesky_factors[k], 
                    logdet[k], farthest.get(k), ml_graph, cl_graph
                )
                costs.append(cost)
            # Assign x^i to the cluster with minimum cost
            new_labels[i] = int(np.argmin(costs))

        return new_labels

    def _assignment_cost(self, X, i, k, labels, mu, Lk, logdet_Mk,
                         farthest_k, ml_graph, cl_graph):          
        """
        This method computes the cost of assigning data point x^i 
        to cluster k, keeping the remaining assignments fixed.
    
        :arg(X, i, k, labels, mu, Lk, logdet_Mk, farthest_k, 
             ml_graph, cl_graph):
            X = data matrix of shape (P, N)
            i = index of the data point to assign
            k = candidate cluster index
            labels = cluster assignments from the previous iteration
            mu = centroid matrix of shape (K, N)
            Lk = Cholesky factor of M_k = A_k^{-1}
            logdet_Mk = log-determinant of M_k
            farthest_k = farthest pair information for cluster k, or None
            ml_graph = Must-Link graph
            cl_graph = Cannot-Link graph
    
        :return: cost = assignment cost for placing x^i in cluster k
        """
        # Distance + log-det term
        d = X[i] - mu[k]
        # − logdet(A_k​) = logdet(M_k​)
        cost = self._mahalanobis_sq(d, Lk) + logdet_Mk 

        # Must-Link violations
        for j in ml_graph[i]:
            if labels[j] != k:
                cost += (
                    0.5 * self.w * self._mahalanobis_sq(X[i] - X[j], Lk)
                    )

        # Cannot-Link violations
        if farthest_k is not None:
            dist_max = farthest_k[2]
            for j in cl_graph[i]:
                if labels[j] == k:
                    dij = self._mahalanobis_sq(X[i] - X[j], Lk)
                    cost += self.w * max(dist_max - dij, 0.0)

        return cost 

    def _repair_empty_clusters(self, X, labels, mu, cholesky_factors):       
        """
        This method prevents clusters from remaining empty after the 
        assignment step.    
        If a cluster is empty, a point is moved from another
        cluster (with at least two points), selecting the one farthest
        from its current centroid under the cluster-specific metric.
    
        :arg(X, labels, mu, cholesky_factors):
            X = data matrix of shape (P, N)
            labels = current cluster assignments 
            mu = centroid matrix of shape (K, N)
            cholesky_factors = Cholesky factors of M_k = A_k^{-1}
    
        :return: labels = updated cluster assignments with 
                          no empty clusters
        """ 
        # number of points per cluster
        counts = np.bincount(labels, minlength = self.K)
        
        # indices of empty clusters
        empty = list(np.where(counts == 0)[0]) 

        for k_empty in empty:
            # find donor clusters with at least 2 points
            donors = np.where(counts > 1)[0]
            if donors.size == 0:
                break
            
            # Moves the point furthest from its current centroid
            best_p = None # index of point moved to empty cluster
            best_k = None # donor cluster from which best_p comes
            best_d = -np.inf
            
            for k in donors:
                # idx are the indices of points in cluster k
                idx = np.where(labels == k)[0]
                for p in idx:
                    d = self._mahalanobis_sq(
                        X[p] - mu[k], cholesky_factors[k])
                    if d > best_d:
                        best_d = d
                        best_p = p
                        best_k = k
            
            if best_p is None:
                break
                      
            # Moving that point to the empty cluster
            labels[best_p] = k_empty
            
            # Update counts consistently
            counts[best_k] -= 1
            counts[k_empty] += 1

        return labels

    def _update_centroids(self, X, labels):       
        """
        This method updates the cluster centroids based on the current
        cluster assignments.
        
        :arg(X, labels):
            X = data matrix of shape (P, N)
            labels = current cluster assignments
    
        :return: mu = updated centroid matrix of shape (K, N)
        """
        mu = np.zeros((self.K, X.shape[1]))
        for k in range(self.K):
            idx = np.where(labels == k)[0]
            mu[k] = X[idx].mean(axis = 0)
        return mu
  
    # METRIC UPDATE
    def _update_metrics(self, X, labels, mu, cholesky_factors, 
                        farthest, ml_graph, cl_graph):
        """
        This method updates the cluster-specific inverse metric
        matrices A_k^{-1}.    
        For each cluster k, the update combines:
            - the intra-cluster scatter term,
            - the Must-Link penalty term,
            - the Cannot-Link hinge-based penalty term.
    
        :arg(X, labels, mu, cholesky_factors, farthest, ml_graph, 
             cl_graph):
            X = data matrix of shape (P, N)
            labels = current cluster assignments 
            mu = centroid matrix of shape (K, N)
            cholesky_factors = Cholesky factors of M_k = A_k^{-1}
            farthest = dictionary of farthest intra-cluster pairs
            ml_graph = Must-Link graph
            cl_graph = Cannot-Link graph
    
        :return: M_new = updated array of A_k^{-1} matrices
        """
        N = X.shape[1]
        M_new = np.zeros((self.K, N, N))
        
        # update one metric at a time
        for k in range(self.K):           
            idx = np.where(labels == k)[0]
            n_k = len(idx) # n_k = |C_k|
            if n_k == 0:
                M_new[k] = np.eye(N)
                continue

            # Scatter term
            scatter = np.zeros((N, N))
            for i in idx:
                v = (X[i] - mu[k]).reshape(-1, 1)
                scatter += v @ v.T

            # Must-Link term
            ml_term = np.zeros((N, N))
            for i in idx:
                for j in ml_graph[i]:
                    if j > i and labels[j] != labels[i]:
                    # j > i helps to avoid counting the same pair twice
                        d = (X[i] - X[j]).reshape(-1, 1)
                        ml_term += 0.5 * self.w * (d @ d.T)

            # Cannot-Link term
            cl_term = np.zeros((N, N))
            # retrieve the farthest pair in cluster k
            far_k = farthest.get(k)
            if far_k is not None:
                i_max, j_max, dist_max = far_k
                u = (X[i_max] - X[j_max]).reshape(-1, 1)
                outer_max = u @ u.T

                for i in idx:
                    for j in cl_graph[i]:
                        if j > i and labels[j] == k:
                            dvec = X[i] - X[j]
                            dist_ij = self._mahalanobis_sq(
                                dvec, cholesky_factors[k])
                            hinge = max(dist_max - dist_ij, 0.0)
                            if hinge > 0:
                                v = dvec.reshape(-1, 1)
                                cl_term += self.w * hinge * (
                                    outer_max - v @ v.T)
                                   
            # A_k^{-1} = (scatter + ML + CL) / |C_k|
            Mk = (scatter + ml_term + cl_term) / n_k

            # Regularize to guarantee SPD
            _, _, Mk_reg = self._regularize_metric(Mk)
            M_new[k] = Mk_reg

        return M_new
