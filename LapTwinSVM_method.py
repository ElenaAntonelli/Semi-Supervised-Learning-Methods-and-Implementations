# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 16:17:55 2025

@author: Elena Antonelli
"""

# Import the libraries
import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from scipy.spatial.distance import cdist


# LapTwinSVM method
class LapTwinSVM(object):
    def __init__(self, opt):
        self.opt = opt.copy() # opt = dictionary with model parameters
        self.verbose = bool(self.opt.get('verbose', False)) 
        
        # Variables 
        self.X = None
        self.alpha_star_pos = None  
        self.theta_star_pos = None  
        self.alpha_star_neg = None  
        self.theta_star_neg = None  

        self._K_train = None  # K over all training data 
        self._A_idx = None    # indices for +1 labeled within X
        self._B_idx = None    # indices for -1 labeled within X

    # Graph Laplacian & kernel functions:
    def _build_graph_laplacian(self, X):       
        '''        
        This method constructs the graph Laplacian from the data X.
        
        :arg(X): 
            X = [X_l; X_u] all data, labeled + unlabeled
            
        : return: Graph Laplacian L        
        '''       
        # Computation of the adjacency matrix W
        mode = self.opt['neighbor_mode']
        # n_nb = number of neighbors to consider in the KNN graph
        n_nb = self.opt['n_neighbor'] 

        if mode == 'connectivity': # unweighted graph
            W = kneighbors_graph(
                X, n_nb, mode='connectivity', include_self=False)
            W = (((W + W.T) > 0) * 1) # makes the matrix symmetric

        elif mode == 'distance':
            # the distance between neighbors is measured with a 
            # specified metric
            W = kneighbors_graph(
                X, n_nb, mode='distance', include_self=False)
            W = W.maximum(W.T)
            # Gaussian weights with heat parameter t
            t = float(self.opt['t'])
            W = sparse.csr_matrix(
                (np.exp(-W.data**2 / (4.0 * t)), W.indices, W.indptr),
                shape=(X.shape[0], X.shape[0])
            )
        else:
            raise ValueError(
                "neighbor_mode must be 'connectivity' or 'distance'"
                )
        
        # Computing Graph Laplacian
        D = sparse.diags(np.array(W.sum(0))[0]).tocsr()
        L = D - W
        return L

    def _kernel(self, X1, X2):
        '''        
        This method computes the kernel matrix K_ij = k(x_i, x_j)  
        between two sets of data points X1,X2.
        
        :arg(X1,X2): 
            X1 = ndarray; first input dataset
            X2 = ndarray; second input dataset
        
        : return: kernel matrix K      
        '''        
        K_matrix = self.opt['kernel_function'](
            X1, X2, **self.opt['kernel_parameters'])       
        return K_matrix

    
    def fit(self, X_l, Y_l, X_u):
        """       
        Train the Laplacian Twin SVM.

        :arg(X_l,Y_l,X_u): 
            X_l = labeled data, 
                  ndarray shape (n_labeled_samples, n_features)
            Y_l = labels of labeled data, 
                  ndarray shape (n_labeled_samples,)
            X_u = unlabeled data, 
                  ndarray shape (n_unlabeled_samples, n_features)               
        """              
        self.X = np.vstack([X_l, X_u]) # X = entire dataset 
        P = self.X.shape[0] # number of rows of the matrix X

        # Indices for labeled points in the stacked array
        n_l = X_l.shape[0] # number of labeled samples
        labeled_idx = np.arange(n_l) 

        # Indices of positive / negative labeled points 
        # (within the stacked array)
        A_mask = (Y_l == 1)
        B_mask = (Y_l == -1)
        A = labeled_idx[A_mask] 
        B = labeled_idx[B_mask] 

        P1 = A.size # total number of elements in A
        P2 = B.size # total number of elements in B
        
        if P1 == 0 or P2 == 0:
            raise ValueError(
                "Both classes must be present in labeled data."
                )

        self._A_idx = A
        self._B_idx = B

        # Graph Laplacian computed over all points 
        L = self._build_graph_laplacian(self.X)

        # Computing K over all points
        K = self._kernel(self.X, self.X)        # (P x P)
        self._K_train = K

        # Submatrices of K with rows only from class A or B
        K_A = K[A, :]                            # (P1 x P)
        K_B = K[B, :]                            # (P2 x P)

        # Regularization weights
        gamma_A = float(self.opt['gamma_A'])
        gamma_I = float(self.opt['gamma_I'])

        # Precompute common pieces for '+' problem:
            
        # Computing H_pos
        # Use sparse L with dense K: (K @ (L @ K)) 
        KL = K @ (L @ K)  # (P x P) 
        H_pos = (K_A.T @ K_A) + gamma_A * K + gamma_I * KL
        
        # e vectors: 1D for optimization; 
        # use [None, :] when a row is needed for stacking
        e_pos = np.ones(P1)                     # (P1,)
        e_neg = np.ones(P2)                     # (P2,)
        
        # Computing q_pos
        q_pos = K_A.T @ e_pos.reshape(-1, 1)    # (P x 1)
               
        # Build Q_pos and T_pos for '+' dual
        Q_pos = np.empty((P + 1, P + 1), dtype=K.dtype)
        Q_pos[:P, :P] = H_pos
        Q_pos[:P, P:P+1] = q_pos
        Q_pos[P:P+1, :P] = q_pos.T
        Q_pos[P, P] = float(P1)

        T_pos = np.vstack([K_B.T, e_neg[None, :]])   # ((P+1) x P2)
               
        # Numerical stabilization for Q_pos 
        eps = float(self.opt.get("ridge_eps", 1e-6))  
        Qp = Q_pos + eps * np.eye(P + 1, dtype=Q_pos.dtype)
        
        # Compute M_pos (without explicit inverse)
        try:
            # Solve the linear system Q v = T  
            # then M = T^T v
            v_pos = np.linalg.solve(Qp, T_pos)       # (P+1 x P2)
        except np.linalg.LinAlgError:           
            v_pos = np.linalg.lstsq(Qp, T_pos, rcond=None)[0]
            # lstsq find the solution with minimum norm    
        
        M_pos = T_pos.T @ v_pos                      # (P2 x P2)
        
        
        # PROBLEM FORMULATION ('+' hyperplane)
        
        # ===== Objectives =====
        def obj_pos(nu):
            nu = np.asarray(nu)
            return 0.5 * (nu @ (M_pos @ nu)) - (e_neg @ nu)
        
        def grad_pos(nu):  # gradient of the objective function
            nu = np.asarray(nu)
            return (M_pos @ nu) - e_neg
        
        # =====Constraint=====
        #   0 <= nu_i <= 1 
        bounds_pos = [(0, 1) for _ in range(P2)]
        
        
        # ===== Solving =====
        nu0 = np.zeros(P2) # initial guess for the parameters

        res_pos = minimize(
            fun=obj_pos,
            x0=nu0,
            jac=grad_pos, # Jacobian
            bounds=bounds_pos,
            method='L-BFGS-B'
        )
        nu_star = res_pos.x # optimal solution of the minimization


        # Recover [alpha*_+ ; theta*_+] 
        z_star_pos = - (v_pos @ nu_star)              # (P+1,)
        self.alpha_star_pos = z_star_pos[:P].reshape(-1, 1)
        self.theta_star_pos = float(z_star_pos[P])


        # PROBLEM FORMULATION ('-' hyperplane)
        
        # Compute common pieces for '-' problem:
        H_neg = (K_B.T @ K_B) + gamma_A * K + gamma_I * KL
        
        q_neg = K_B.T @ e_neg.reshape(-1, 1)
        
        Q_neg = np.empty((P + 1, P + 1), dtype=K.dtype)
        Q_neg[:P, :P] = H_neg
        Q_neg[:P, P:P+1] = q_neg
        Q_neg[P:P+1, :P] = q_neg.T
        Q_neg[P, P] = float(P2)
       
        T_neg = np.vstack([K_A.T, e_pos[None, :]])   # ((P+1) x P1)
        
        # Numerical stabilization for Q_neg 
        eps = float(self.opt.get("ridge_eps", 1e-6))
        Qn = Q_neg + eps * np.eye(P + 1, dtype=Q_neg.dtype)
        
        try:
            v_neg = np.linalg.solve(Qn, T_neg)       # (P+1 x P1)
        except np.linalg.LinAlgError:
            v_neg = np.linalg.lstsq(Qn, T_neg, rcond=None)[0]
        
        M_neg = T_neg.T @ v_neg                      # (P1 x P1)
        
        # ===== Objectives =====
        def obj_neg(mu):
            mu = np.asarray(mu)
            return 0.5 * (mu @ (M_neg @ mu)) - (e_pos @ mu)
       
        def grad_neg(mu):
            mu = np.asarray(mu)
            return (M_neg @ mu) - e_pos
        
        # =====Constraint=====
        bounds_neg = [(0, 1) for _ in range(P1)]
        
        
        # ===== Solving =====
        mu0 = np.zeros(P1)

        res_neg = minimize(
            fun = obj_neg,
            x0 = mu0,
            jac = grad_neg,
            bounds=bounds_neg,
            method='L-BFGS-B'
        )
        mu_star = res_neg.x
        
        # Recover [alpha*_- ; theta*_-]
        z_star_neg = v_neg @ mu_star
        self.alpha_star_neg = z_star_neg[:P].reshape(-1, 1)
        self.theta_star_neg = float(z_star_neg[P])
        
        
        # Reports whether the algorithm converged or not
        if self.verbose:
            print("Optimization '+' success:",
                  res_pos.success, res_pos.message)
            print("Optimization '-' success:",
                  res_neg.success, res_neg.message)
        return self
   
    def _decision_pair(self, Xtest):
        """       
        This method compute (f_+(x), f_-(x)) for each test point.
        
        :arg(Xtest): 
            Xtest = test data, ndarray
        
        :return: fpos = ndarray, shape (n_test,)
                 fneg = ndarray, shape (n_test,)           
        """       
        if self.X is None:
            raise RuntimeError("Model not fitted.")
        
        # Computing K over test points by calling the function _kernel
        K_tx = self._kernel(self.X, Xtest)            # (P x n_test)

        fpos = (
            (self.alpha_star_pos.T @ K_tx).ravel() 
            + self.theta_star_pos)
        fneg = (
            (self.alpha_star_neg.T @ K_tx).ravel() 
            + self.theta_star_neg)
        return fpos, fneg


    def decision_function(self, Xtest):
        """        
        Function that compute a decision score for each test sample 
        based on the relative distance to the two twin hyperplanes.
        A positive score indicates classification as +1,
        a negative score indicates classification as -1.
        
        :arg(Xtest):
            Xtest = test samples
            
        :return: decision score  d_-(x) - d_+(x)          
        """
        fpos, fneg = self._decision_pair(Xtest)

        # ||w|| = sqrt(alpha^T K alpha), K over all training + unlabeled
        K = self._K_train
        eps = 1e-12
        nrm_pos = float(np.sqrt(
            self.alpha_star_pos.T @ K @ self.alpha_star_pos
            ) + eps)
        nrm_neg = float(np.sqrt(
            self.alpha_star_neg.T @ K @ self.alpha_star_neg
            ) + eps)
        # 1e-12 to stabilize the denominator and prevent
        # divisions or zero roots
        
        # Compute distances to each hyperplane
        dpos = np.abs(fpos) / nrm_pos
        dneg = np.abs(fneg) / nrm_neg
       
        return dneg - dpos 
               # positive score means closer to '+' hyperplane

    def predict(self, Xtest):
        '''       
        This method predicts the label for each point in Xtest using 
        decision_function.
        
        :arg(Xtest): 
            Xtest = test data, ndarray shape (n_samples, n_features)
            
        : return: labels {-1,1}  
        '''
        score = self.decision_function(Xtest)
        return np.where(score > 0, 1, -1)


    def accuracy(self, Xtest, Ytrue):
        """       
        Method that calculates the accuracy of predictions. 
        
        :arg(Xtest, Ytrue):
            Xtest = test data, ndarray shape (n_samples, n_features)
            Ytrue = test labels, ndarray shape (n_samples, )
        
        : return: accuracy             
        """
        pred = self.predict(Xtest)
        acc = np.mean(pred == Ytrue)
        accuracy_percent = round(float(acc) * 100.0, 2)
        return accuracy_percent 



def rbf(X1, X2, **kwargs):
    """   
    Function that calculates the Gaussian Kernel. 
    
    :arg(X1, X2, **kwargs):
        X1, X2 = two sets of samples
        **kwargs = dictionary that collects all the optional parameters  
                   passed to the function via keyword.
                   Inside kwargs there is a key-value pair where:
                    - Keys are the names of the arguments 
                    - Values ​​are the associated values 
    
    : return: gaussian Kernel K(x, y) = exp(-gamma ||x-y||^2)
        
    """
    gamma = float(kwargs['gamma'])
    return np.exp(-cdist(X1, X2)**2 * gamma) # cdist = Euclidean distance
