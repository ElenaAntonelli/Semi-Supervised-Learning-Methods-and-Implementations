# -*- coding: utf-8 -*-
"""
Created on Tue May  6 16:16:56 2025

@author: Elena Antonelli
"""


# Import the libraries
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from scipy import sparse


# Variables Initialization 
Q = None


# LapSVM method
class LapSVM(object):
    def __init__(self,opt):
        self.opt = opt # opt = dictionary with model parameters

    def fit(self,X_l,Y_l,X_u): 
        '''
        This method is used to train the LapSVM model.
        
        :arg(X_l,Y_l,X_u): 
            X_l = labeled data, 
                  ndarray shape (n_labeled_samples, n_features)
            Y_l = labels of labeled data, 
                  ndarray shape (n_labeled_samples,)
            X_u = unlabeled data, 
                  ndarray shape (n_unlabeled_samples, n_features)  
        '''
        
        # GRAPH CONSTRUCTION
        
        self.X = np.vstack([X_l,X_u]) 
        Y = np.diag(Y_l) 
        
        # Computation of the adjacency matrix W
        if self.opt['neighbor_mode'] == 'connectivity': 
        # unweighted graph
            
            # use of knn for graph construction
            W = kneighbors_graph(self.X, self.opt['n_neighbor'],
                                 mode='connectivity',
                                 include_self=False) 
            W = (((W + W.T) > 0) * 1) # makes the matrix symmetric
                     
        elif self.opt['neighbor_mode'] == 'distance':
            # mode = 'distance' will return the distances between 
            # neighbors according to the given metric
            # (Minkowski by default)
            W = kneighbors_graph(self.X, self.opt['n_neighbor'],
                                 mode = 'distance',include_self = False)
            W = W.maximum(W.T) # Symmetrize by assigning Wij = Wji 
                               # the maximum between the two
            
            # Apply Gaussian weights 
            W = sparse.csr_matrix((np.exp(-W.data**2/4/self.opt['t']),
                                   W.indices, W.indptr),
                                   shape=(self.X.shape[0],
                                   self.X.shape[0])) 
            # W is in CSR (Compressed Sparse Row) format
        else:
            raise Exception()

    
        # Computing Graph Laplacian
        D = sparse.diags(np.array(W.sum(0))[0]).tocsr() 
        # tocsr = to csr; converts a sparse matrix to CSR format
        
        L = D - W 
            
        
        # Computing K with k(i,j) = kernel(i, j)
        K = self.opt['kernel_function'](self.X,self.X,
                                        **self.opt['kernel_parameters'])
        
        
        # Matrices preparation for dual formulation
        l = X_l.shape[0] # l = P'
        u = X_u.shape[0] # u = P - P'
        
        # Creating matrix J [I (l x l), 0 (l x u)]
        J = np.concatenate([np.identity(l), 
                            np.zeros(l * u).reshape(l, u)], axis=1)
        
        # Computing "almost" alpha (without beta)
        almost_alpha = np.linalg.inv(2 * self.opt['gamma_A'] * 
                                     np.identity(l + u) \
                                     + (2 * self.opt['gamma_I']) *
                                     L.dot(K)).dot(J.T).dot(Y)
        
        # Computing Q
        Q = Y.dot(J).dot(K).dot(almost_alpha)
        Q = (Q+Q.T)/2 # symmetrization
       
        del W, L, K, J

        e = np.ones(l)
        q = -e
        
        
        # PROBLEM FORMULATION
        
        # ===== Objectives =====
        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)

        def objective_grad(beta): # gradient of the objective function
            return np.squeeze(np.array(beta.T.dot(Q) + q))
        # np.squeeze(...) removes any dimensions of length 1 

        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 
        bounds = [(0, 1) for _ in range(l)]

        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(beta):
            return beta.dot(Y_l)

        def constraint_grad(beta):
            return Y_l

        cons = {'type': 'eq', 'fun': constraint_func, 
                'jac': constraint_grad}

        # ===== Solving =====
        x0 = np.zeros(l)

        beta_star = minimize(objective_func, x0,
                            jac=objective_grad, constraints=cons,
                            bounds=bounds)['x']
        # x0: Initial guess for the parameters;
        # jac = Jacobian
        
        
        # Computing final alpha
        self.alpha_star = almost_alpha.dot(beta_star)

        del almost_alpha, Q

        # Estimating the bias term (theta*) using labeled data
        new_K = self.opt['kernel_function'](self.X,X_l,
                                **self.opt['kernel_parameters'])
        f = np.squeeze(np.array(self.alpha_star)).dot(new_K)
        
        # Identifying Support Vector indices
        self.sv_ind=np.nonzero((beta_star>1e-7)*(beta_star<(1-1e-7)))[0] 
        
       
        if len(self.sv_ind) > 0: # if there are support vectors
            # only the elements of Y and f corresponding to the
            # support vectors are considered
            ys = np.diag(Y)[self.sv_ind] 
            fs = f[self.sv_ind]
            self.theta_star = np.mean(ys - fs)
           
        else:
            print(
            'Info: No support vectors found in the range 0 < beta < 1.'
                )
            print(
                "Proceeding with alternative bias estimation "
                "using near-margin labeled points."
                )
            
            if len(Y_l) == 1:
                self.theta_star = 0               
            else:
                # Take maximum 60% of the labeled, but at least 2 
                k = max(2, int(0.6 * len(Y_l)))
                
            
            # estimate theta* from a set of "near-edge" points          
            margin_vals = np.abs(Y_l * f - 1)
            # Note! y(i)*f(i) > 0 if the point i is correctly classified
            #       y(i)*f(i) = 1 if the point is exactly on the margin
            
                          
            # takes the indices of the first k points closest to the 
            # margin (i.e. those with smaller margin_vals) and 
            # rearranges them from closest to furthest
            close_points = np.argsort(margin_vals)[:k]
            
            theta_vals = [Y_l[i] - f[i] for i in close_points]
            self.theta_star = np.mean(theta_vals) 
           
        
    def decision_function(self,Xtest):
        '''
        This method computes the decision function f(x) + theta 
        for each point in X. The obtained values ​​are not yet 
        classified labels, but real values.
        
        :arg(Xtest): 
            Xtest = test data, ndarray shape (n_samples, n_features)
            
        : return: f(x) = sum [ alpha_i * k(x_i,x) ] + theta
        '''             
        new_K = self.opt['kernel_function'](self.X, Xtest,
                                     **self.opt['kernel_parameters'])
        f = np.squeeze(np.array(self.alpha_star)).dot(new_K)
        return f + self.theta_star   
        

    def predict(self, Xtest):
        '''
        This method predicts the label for each point in Xtest using 
        the decision function.
        
        :arg(Xtest): 
            Xtest = test data, ndarray shape (n_samples, n_features)
            
        : return: labels {-1,1}    
        '''      
        f = self.decision_function(Xtest)
        return np.sign(f) 
    
    
    def accuracy(self, Xtest, Ytrue):
        """
        Method that calculates the accuracy of predictions. 
        
        :arg(Xtest, Ytrue):
            Xtest = test data, ndarray shape (n_samples, n_features)
            Ytrue = test labels, ndarray shape (n_samples, )
        
        : return: accuracy           
        """
        predictions = self.predict(Xtest)
        accuracy = sum(predictions == Ytrue) / len(predictions)
        accuracy_percent = round(accuracy * 100, 2)
        return  accuracy_percent


def rbf(X1,X2,**kwargs):
    """
    Method that calculates the Gaussian Kernel. 
    
    :arg(X1, X2, **kwargs):
        X1, X2 = two sets of samples
        **kwargs = dictionary that collects all the optional parameters passed 
                   to the function via keyword.
                   Inside kwargs there is a key-value pair where:
                    - Keys are the names of the arguments 
                    - Values ​​are the associated values 
    
    : return: gaussian Kernel K(x, y) = exp(-gamma ||x-y||^2)       
    """
    return np.exp(-cdist(X1,X2)**2*kwargs['gamma'])  # cdist = Euclidean distance
   
    


