# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 09:43:21 2025

@author: Elena Antonelli

"""

# import libraries
import math
import sys

import numpy as np
from scipy import optimize
from scipy import sparse


def _as_2d_array(x, dtype = np.float64):
    """
    This function converts the input data into a 2D NumPy array.
    
    :arg(x, dtype):
        x = input data, provided either as a vector or as a matrix
        dtype = the desired numeric type of the output array 
    
    : return: a 2D NumPy array of shape (n_samples, n_features)
    """   
    a = np.asarray(x, dtype = dtype)
    if a.ndim == 1: # a = single data point (vector)
        a = a.reshape(1, -1) # convert to one-row matrix
    elif a.ndim != 2: # higher-dimensional input
        a = a.reshape(a.shape[0], -1)
    return a


def _col(x, dtype = np.float64):
    """
    Function that converts the input data into 
    a column vector (n,1) ndarray.
    
    :arg(x, dtype):
        x = input data representing a vector
        dtype = the desired numeric type of the output array 
    
    : return: a column vector of shape (n_samples, 1).
    """    
    a = np.asarray(x, dtype=dtype).reshape(-1, 1)
    return a


class QN_S3VM:
    """
    L-BFGS optimizer for semi-supervised support vector machines (S3VM).
    """

    def __init__(self, X_l, L_l, X_u, random_generator = None, **kw):
        """
        Initializes the model. 
        Detects automatically if dense or sparse data is provided.

        Keyword arguments:
        X_l = samples of labeled part of the data
        L_l = labels of labeled part of the data
        X_u = samples of unlabeled part of the data
        random_generator = particular instance of a random generator
        kw = additional parameters for the optimizer
        """
        self.__model = None # no model chosen yet

        # assign the internal model implementation
        if sparse.issparse(X_l):
            self.__model = QN_S3VM_Sparse(X_l, L_l, X_u, 
                                          random_generator, **kw)
        else:
            self.__model = QN_S3VM_Dense(X_l, L_l, X_u, 
                                         random_generator, **kw)
        
        # Data format unknown
        if self.__model is None:
            raise TypeError("Data format for patterns is unknown.")

    def train(self):
        """
        Training phase.

        : return: list of predicted labels for the training set
        """
        return self.__model.train()

    def getPredictions(self, X, real_valued = False):
        """
        This function computes the predicted labels for a given set of samples.
        
        :arg(X, real_valued): 
            X = input data matrix         
            real_valued = if True, then the real prediction values 
                          are returned
            
        : return: list of predictions for X 
        """
        return self.__model.getPredictions(X, real_valued=real_valued)

    def predict(self, x):
        """
        Method that predicts a label (-1 or +1) for a given input sample.

        :arg(x): 
            x = input sample
        
        : return: the prediction for x
        """
        return self.__model.predict(x)

    def predictValue(self, x):
        """
        Computes f(x) for a given sample (Representer Theorem).
    
        :arg(x): 
            x = input sample 

         : return: the (real) prediction value for x
        """
        return self.__model.predictValue(x)
    
    def accuracy(self, X, y): 
        """
        Method that calculates the accuracy of the model predictions. 
        
        :arg(X, y):
            X = input data matrix
            y = true class labels associated with X
        
        : return: accuracy expressed as a percentage          
        """
        pred = self.getPredictions(X, real_valued=False)
        acc = np.mean(pred == y)
        accuracy_percent = round(float(acc) * 100.0, 2)
        return accuracy_percent


    def getNeededFunctionCalls(self):
        """
        Function that returns the number of function calls needed during 
        the optimization process.      
        """
        return self.__model.getNeededFunctionCalls()



# Kernels 

class LinearKernel:
    """
    Linear Kernel
    """
    def computeKernelMatrix(self, data1, data2, symmetric = False):
        """
        This function computes the kernel matrix.
       
        :arg(data1, data2, symmetric):
            data1 = first input data matrix
            data2 = second input data matrix
            symmetric = boolean flag indicating whether the resulting 
                    kernel matrix is expected to be symmetric          
        
        : return: kernel matrix        
        """
        X1 = _as_2d_array(data1)
        X2 = _as_2d_array(data2)
        if X1.shape[1] != X2.shape[1]:
            raise ValueError(
                f"LinearKernel: dim mismatch {X1.shape} vs {X2.shape}"
                )
        return X1 @ X2.T

    def getKernelValue(self, xi, xj):
        """
        Returns a single kernel value.
        
        :arg(xi, xj):
            xi = first input sample
            xj = second input sample
            
        : return: kernel value k(xi, xj) = xi^T xj             
        """
        xi = np.asarray(xi, dtype=np.float64)
        xj = np.asarray(xj, dtype=np.float64)
        return float(np.dot(xi, xj))


class RBFKernel:
    """
    RBF Kernel: K(x,z) = exp(-||x-z||^2 / (2 sigma^2))
    """
    def __init__(self, sigma):
        self.sigma = float(sigma)

    def computeKernelMatrix(self, data1, data2, symmetric = False):
        """
        This function computes the kernel matrix.
       
        :arg(data1, data2, symmetric):
            data1 = first input data matrix
            data2 = second input data matrix
            symmetric = boolean flag indicating whether the resulting
                        kernel matrix is expected to be symmetric         
        
        : return: kernel matrix        
        """
        X1 = _as_2d_array(data1)
        X2 = _as_2d_array(data2)
        if X1.shape[1] != X2.shape[1]:
            raise ValueError(
                f"RBFKernel: dim mismatch {X1.shape} vs {X2.shape}"
                )
        # squared Euclidean distances
        # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x*y
        x1_sq = np.sum(X1 * X1, axis=1, keepdims=True)  
        x2_sq = np.sum(X2 * X2, axis=1, keepdims=True).T  
        d2 = x1_sq + x2_sq - 2.0 * (X1 @ X2.T)
        gamma = 1.0 / (2.0 * (self.sigma ** 2))
        return np.exp(-gamma * d2)

    def getKernelValue(self, xi, xj):
        """
        Returns a single kernel value.
        
        :arg(xi, xj):
            xi = first input sample
            xj = second input sample
            
        : return: kernel value k(xi, xj)         
        """
        xi = np.asarray(xi, dtype=np.float64)
        xj = np.asarray(xj, dtype=np.float64)
        d2 = float(np.sum((xi - xj) ** 2))
        gamma = 1.0 / (2.0 * (self.sigma ** 2))
        return float(math.exp(-gamma * d2))



# Dense implementation

class QN_S3VM_Dense:
    """
    L-BFGS optimizer for S3VM with dense data.
    """
    
    parameters = {
        "lam": 1.0,   # regularization parameter (lambda > 0) 
        "lamU": 1.0,  # cost parameter that determines influence of
                      # unlabeled samples; float > 0 
        "sigma": 1.0, #  kernel width for RBF kernel
        "kernel_type": "Linear",  # "Linear" or "RBF"
        
        "numR": None, # implementation of subset of regressors 
                      # if None, all samples are used
        # Must fulfill 0 <= numR <= len(X_l) + len(X_u) 
        "estimate_r": None, # desired ratio for positive and negative 
                            # assigments for unlabeled samples
                            # (-1.0 <= estimate_r <= 1.0)
        # If estimate_r=None, then L_l is used to estimate this ratio
        # (in case len(L_l) >= minimum_labeled_patterns_for_estimate_r. 
        # Otherwise use estimate_r = 0.0
        "minimum_labeled_patterns_for_estimate_r": 0,
        
        "BFGS_m": 50,       # BFGS parameter
        "BFGS_maxfun": 500, # BFGS parameter, maximum number of function calls
        "BFGS_factr": 1e12, # BFGS parameter
        "BFGS_pgtol": 1.0000000000000001e-05, #  BFGS parameter
        "BFGS_verbose": -1, # verbosity level of the L-BFGS optimizer
        
        "surrogate_s": 3.0,          # scaling parameter
        "surrogate_gamma": 20.0,     # scaling factor
        "breakpoint_for_exp": 500.0, # numerical stability threshold
    }

    def __init__(self, X_l, L_l, X_u, random_generator, **kw):
        """
        Intializes the S3VM optimizer.
        """
        if random_generator is None:
            self.__random_generator = np.random.default_rng()
        elif isinstance(random_generator, (int, np.integer)):
            self.__random_generator = np.random.default_rng(
                int(random_generator))
        elif isinstance(random_generator, np.random.Generator):
            self.__random_generator = random_generator
        else:
            # ignore unsupported RNG objects and use a fresh Generator
            self.__random_generator = np.random.default_rng()
            
        self.__X_l = _as_2d_array(X_l)
        self.__X_u = _as_2d_array(X_u)
        self.__L_l = np.asarray(L_l, dtype=np.float64).reshape(-1)
        assert self.__X_l.shape[0] == self.__L_l.shape[0]
        # assert = to check if a condition is true:
        # if true nothing happens, 
        # otherwise Python throws an exception and the program stops

        # full data matrix
        self.__X = np.vstack([self.__X_l, self.__X_u])  
        self.__size_l = self.__X_l.shape[0]
        self.__size_u = self.__X_u.shape[0]
        self.__size_n = self.__size_l + self.__size_u

        self.__matrices_initialized = False
        self.__needed_function_calls = 0
        self.__setParameters(**kw)
        # **kw : pass an arbitrary number of named parameters,
        #        collected in a dictionary
        self.__kw = kw

    def train(self):
        """
        Training phase.

        : return: list of predicted labels for the full training set
        """
        indi_opt = self.__optimize()
        self.__recomputeModel(indi_opt)
        predictions = self.__getTrainingPredictions()
        return predictions

    def getPredictions(self, X, real_valued = False):
        """
        Function that computes the predicted labels for a given set of samples.

        :arg(X, real_valued): 
            X = input data matrix         
            real_valued = if True, then the real-valued 
                        prediction are returned
            
        : return: list of predictions for X        
        """
        # Only the kernel terms that depend on X are recomputed,
        # while the centering statistics, computed on the 
        # unlabeled subset, are kept fixed
        
        X = _as_2d_array(X)
        KNR = self.__kernel.computeKernelMatrix(
            X, self.__Xreg, symmetric=False)  # (m, numR)
        KNU_bar = self.__kernel.computeKernelMatrix(
            X, self.__X_u_subset, symmetric=False)  # (m, |Ubar|)
        KNU_bar_horizontal_sum = (
            1.0 / self.__X_u_subset.shape[0]
            ) * KNU_bar.sum(axis=1, keepdims=True)  # (m,1)
        # KNU_bar_horizontal_sum = vector containing, 
        # for each data point x_i, the mean kernel value 
        # with respect to the unlabeled subset

        
        KNR = (
            KNR 
            - KNU_bar_horizontal_sum 
            - self.__KU_barR_vertical_sum 
            + self.__KU_barU_bar_sum
            )

        preds = (KNR @ self.__c[:-1]) + self.__c[-1]  # (m,1)
        preds = preds.reshape(-1)

        if real_valued:
            return preds.tolist()
        return np.sign(np.sign(preds) + 0.1).astype(int).tolist()

    def predict(self, x):
        """
        Predicts a label for a given input sample.

        :arg(x): 
            x = input sample
        
        : return: the prediction for x
        """
        return self.getPredictions([x], real_valued=False)[0]

    def predictValue(self, x):
        """
        Computes f(x) for a given input sample (Representer Theorem).
    
        :arg(x): 
            x = input sample

        : return: the (real) prediction value for x.
        """
        return self.getPredictions([x], real_valued=True)[0]

    def getNeededFunctionCalls(self):
        """
        Returns the number of function calls needed during 
        the optimization process.
        """
        return int(self.__needed_function_calls)


    def __setParameters(self, **kw):
        """
        Sets the model hyperparameters by merging user-provided 
        values with default settings and stores the final 
        configuration into internal variables.
    
        :arg (**kw): 
            kw = dictionary of user-defined parameter overrides     
        """       
        self.parameters = dict(self.parameters)
        # self.parameters contains all the default settings
        self.parameters.update(kw) 

        self.__lam = float(self.parameters["lam"])
        assert self.__lam > 0
        self.__lamU = float(self.parameters["lamU"])
        assert self.__lamU > 0
        self.__lam_Uvec = [self.__lamU * i for i in [
            0, 0.000001, 0.0001, 0.01, 0.1, 0.5, 1.0]]
        # the model will be trained several times,
        # with increasing weights on the unlabeled       
        self.__sigma = float(self.parameters["sigma"])
        assert self.__sigma > 0
        
        self.__kernel_type = str(self.parameters["kernel_type"])

        if self.parameters["numR"] is not None:
            self.__numR = int(self.parameters["numR"])
            assert 0 <= self.__numR <= self.__size_n
        else:
            self.__numR = self.__size_n

        self.__dim = self.__numR + 1  # coefficients + bias

        self.__minimum_labeled_patterns_for_estimate_r = float(
            self.parameters["minimum_labeled_patterns_for_estimate_r"])
        if self.parameters["estimate_r"] is not None:
            self.__estimate_r = float(self.parameters["estimate_r"])
        elif len(self.__L_l) >= self.__minimum_labeled_patterns_for_estimate_r:
            self.__estimate_r = float(np.mean(self.__L_l))
        else:
            self.__estimate_r = 0.0

        self.__BFGS_m = int(self.parameters["BFGS_m"])
        self.__BFGS_maxfun = int(self.parameters["BFGS_maxfun"])
        self.__BFGS_factr = float(self.parameters["BFGS_factr"])
        
        # This is a hack for 64 bit systems. The machine precision 
        # is different for the BFGS optimizer and this is fixed by:
        if sys.maxsize > 2**32:
            self.__BFGS_factr = 0.000488288 * self.__BFGS_factr
                       
        self.__BFGS_pgtol = float(self.parameters["BFGS_pgtol"])
        self.__BFGS_verbose = int(self.parameters["BFGS_verbose"])

        self.__surrogate_gamma = float(self.parameters["surrogate_gamma"])
        self.__s = float(self.parameters["surrogate_s"])
        self.__breakpoint_for_exp = float(
            self.parameters["breakpoint_for_exp"])

        self.__b = float(self.__estimate_r)
        self.__max_unlabeled_subset_size = 1000

        # choose regressors indices
        if self.__numR == self.__size_n:
            self.__regressors_indices = np.arange(self.__size_n, dtype=int) 
        else:
            r = int(min(max(self.__numR, 0), self.__size_n))
            if r == 0:
                self.__regressors_indices = np.array([], dtype=int)
            else:
                self.__regressors_indices = np.sort(
                    self.__random_generator.choice(
                        self.__size_n, size=r, replace=False).astype(int)
                )

    def __optimize(self):
        """
        Runs the main optimization procedure. 
    
        : return: (c_current, f_opt), where c_current is the 
                 optimized parameter vector (including bias)
                 and f_opt is the final objective function value
        """
        # Resetting the internal counter used to monitor 
        # how many times the loss is calculated
        self.__needed_function_calls = 0
        self.__initializeMatrices()
        
        # starting point c=[0,0,...,0,b]
        c_current = np.zeros(self.__dim, dtype=np.float64)
        c_current[-1] = self.__b
        
        
        # Annealing sequence.
        # Annealing = starting with an easier version of the problem
        # and gradually making it more difficult
        for lamU in self.__lam_Uvec: 
        # every solution is the starting point for the next step
            self.__lamU = float(lamU)
            
            # optimize only coefficients, keep bias fixed
            c_current = c_current[:-1] 
            c_current = self.__localSearch(c_current)
            c_current = np.append(c_current, self.__b)

        # minimum value of the objective function
        f_opt = self.__getFitness(c_current) 
        return c_current, f_opt

    def __localSearch(self, start):       
        """
        Performs a local optimization step using the L-BFGS algorithm,
        minimizing the surrogate objective with an explicit gradient.
    
        :arg(start):
            start = initial vector of kernel coefficients
        
        : return: optimized vector of kernel coefficients
        """

        c_opt, f_opt, d = optimize.fmin_l_bfgs_b(
            self.__getFitness, # function to minimize
            start,                     # starting point
            m=self.__BFGS_m, # memory used to approximate the Hessian
            fprime=self.__getFitness_Prime, # gradient of the function
            maxfun=self.__BFGS_maxfun, # maximum number of ratings
            factr=self.__BFGS_factr,   # convergence criterion
            pgtol=self.__BFGS_pgtol,   # convergence criterion on the gradient
            iprint=self.__BFGS_verbose # print level   
        )
        
        self.__needed_function_calls += int(d.get("funcalls", 0))
        return np.asarray(c_opt, dtype=np.float64) 
               # c_opt = vector of optimized coefficients

    def __initializeMatrices(self):
        """
        Initializes and stores the kernel-related matrices
        required for the objective and gradient evaluations.
        """
        if self.__matrices_initialized:
            return

        # labels as (n_l,1)
        self.__YL = _col(self.__L_l, dtype=np.float64)

        # kernel
        if self.__kernel_type == "Linear":
            self.__kernel = LinearKernel()
        elif self.__kernel_type == "RBF":
            self.__kernel = RBFKernel(self.__sigma)
        else:
            raise ValueError(f"Unknown kernel_type: {self.__kernel_type}")

        # regressors
        self.__Xreg = self.__X[self.__regressors_indices, :]  # (numR, d)

        # kernel blocks
        self.__KLR = self.__kernel.computeKernelMatrix(
            self.__X_l, self.__Xreg, symmetric=False)  # (l, R)
        self.__KUR = self.__kernel.computeKernelMatrix(
            self.__X_u, self.__Xreg, symmetric=False)  # (u, R)
        self.__KNR = np.vstack([self.__KLR, self.__KUR])  # (n, R)
        self.__KRR = self.__KNR[self.__regressors_indices, :]  # (R, R)

        # subset of unlabeled for centering
        subset_size = min(self.__max_unlabeled_subset_size, self.__size_u)
        s = int(min(max(subset_size, 0), self.__size_u))
        if s == 0:
            subset_unlabeled_indices = np.array([], dtype=int)
        else:
            subset_unlabeled_indices = np.sort(
                self.__random_generator.choice(
                    self.__size_u, size=s, replace=False).astype(int)
            )
        self.__X_u_subset = self.__X_u[subset_unlabeled_indices, :]  # (|Ubar|, d)

        # compute centering terms
        KNU_bar = self.__kernel.computeKernelMatrix(
            self.__X, self.__X_u_subset, symmetric=False)  # (n,|Ubar|)
        KNU_bar_horizontal_sum = (1.0 / self.__X_u_subset.shape[0]
                                  ) * KNU_bar.sum(axis=1, keepdims=True)  # (n,1)       
        
        KU_barR = self.__kernel.computeKernelMatrix(
            self.__X_u_subset, self.__Xreg, symmetric=False)  # (|Ubar|,R)
        self.__KU_barR_vertical_sum = (1.0 / self.__X_u_subset.shape[0]
                                       ) * KU_barR.sum(axis=0, keepdims=True)  # (1,R)

        KU_barU_bar = self.__kernel.computeKernelMatrix(
            self.__X_u_subset, self.__X_u_subset, symmetric=False)
        self.__KU_barU_bar_sum = float(
            (1.0 / (self.__X_u_subset.shape[0] ** 2)
             ) * KU_barU_bar.sum()
            )

        # center KNR and derive blocks again
        self.__KNR = (
            self.__KNR 
            - KNU_bar_horizontal_sum 
            - self.__KU_barR_vertical_sum 
            + self.__KU_barU_bar_sum
            )
        self.__KRR = self.__KNR[self.__regressors_indices, :]
        self.__KLR = self.__KNR[: self.__size_l, :]
        self.__KUR = self.__KNR[self.__size_l :, :]

        self.__matrices_initialized = True

    def __recomputeModel(self, indi):      
        """
        Stores the optimized parameter vector returned by the optimizer 
        into the internal model representation.
    
        :arg(indi):
            indi = vector of kernel coefficients optimized by the L-BFGS algorithm
        """
        # indi[0] = vector of optimal coefficients c_opt found by L-BFGS
        self.__c = _col(indi[0], dtype=np.float64) 

    def __getTrainingPredictions(self, real_valued = False):      
        """
        Computes model predictions on the training set using 
        the optimized coefficients. 
        If real_valued is False, returns class labels in {-1, +1};
        otherwise returns the decision values f(x).
    
        :arg(real_valued):          
            real_valued = if True, returns real-valued decision scores;
                          if False, returns class labels
        
        : return: list of predictions for the training samples 
                 (real-valued scores or {-1, +1} labels)
        """
        preds = (self.__KNR @ self.__c[:-1]) + self.__c[-1]  # (n,1)
        preds = preds.reshape(-1)
        if real_valued:
            return preds.tolist()
        return np.sign(np.sign(preds) + 0.1).astype(int).tolist()
      
    def __getFitness(self, c):          
        """
        Compute the objective function of S3VM. The objective combines:
        (i) a smooth surrogate loss on labeled samples; 
        (ii) an unlabeled penalty term;
        (iii) an RKHS regularization term.
    
        :arg(c):
            c = parameter vector;
                coefficients, optionally including the bias term
        
        : return: scalar value of the surrogate objective function
        """
        # Check whether the function is called from the bfgs solver 
        # (that does not optimize the offset b) or not
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        if c.shape[0] == self.__dim - 1:
            c = np.append(c, self.__b)

        b = float(c[-1])
        c_new = c[:-1].reshape(-1, 1)  

        preds_labeled = self.__surrogate_gamma * (
            1.0 - (self.__YL * ((self.__KLR @ c_new) + b)))
        preds_unlabeled = (self.__KUR @ c_new) + b
        
        # Goal: handle cases where preds_labeled is very large
        conflict = np.sign(np.sign(
            preds_labeled / self.__breakpoint_for_exp - 1.0) + 1.0)
        # conflict has a one for each "numerically instable" entry; 
        # zeros for "good ones"
        good = -1.0 * (conflict - 1.0) 
        # good has a one for each good entry and zero otherwise
        preds_labeled_for_conflicts = conflict * preds_labeled
        preds_labeled = preds_labeled * good
        
        
        # Smooth approximation of hinge loss:
            
        # Compute values for good entries
        preds_labeled_log_exp = np.log(1.0 + np.exp(preds_labeled))
        # Compute values for instable entries
        preds_labeled_log_exp = good * preds_labeled_log_exp
        
        # Replace critical values 
        # Note! For unstable values ​​log(1+e^z) is approximated by z
        preds_labeled_final = (
            preds_labeled_log_exp + preds_labeled_for_conflicts
            )
        
        # Average loss on labeled
        term1 = (1.0 / (self.__surrogate_gamma * self.__size_l)
                 ) * float(np.sum(preds_labeled_final))  
        
        # Note! Using exp(−sf(x)^2) instead of the theoretical 
        # quantity (1-f(x)) because L-BFGS requires a smooth function
        preds_unlabeled_squared = preds_unlabeled * preds_unlabeled
        term2 = (float(self.__lamU) / float(self.__size_u)
                 ) * float(np.sum(np.exp(
                     -self.__s * preds_unlabeled_squared
                     )))

        term3 = float(self.__lam * (c_new.T @ self.__KRR @ c_new))

        return term1 + term2 + term3

    def __getFitness_Prime(self, c):       
        """
        Computes the gradient of the surrogate objective function
        with respect to the model coefficients.
        This gradient is provided to the L-BFGS optimizer.
    
        :arg(c):
            c = parameter vector;
                coefficients, optionally including the bias term
        
        : return: gradient vector with respect to the coefficients 
                  (bias excluded)
        """             
        # Check whether the function is called from the bfgs solver 
        # (that does not optimize the offset b) or not
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        if c.shape[0] == self.__dim - 1:
            c = np.append(c, self.__b)

        b = float(c[-1])
        c_new = c[:-1].reshape(-1, 1)  

        preds_labeled = self.__surrogate_gamma * (
            1.0 - (self.__YL * ((self.__KLR @ c_new) + b)))
        preds_unlabeled = (self.__KUR @ c_new) + b

        conflict = np.sign(np.sign(
            preds_labeled / self.__breakpoint_for_exp - 1.0) + 1.0)
        good = -1.0 * (conflict - 1.0)

        preds_labeled = preds_labeled * good
        preds_labeled_exp = np.exp(preds_labeled)

        term1 = preds_labeled_exp * (1.0 / (1.0 + preds_labeled_exp))
        term1 = good * term1
        term1 = term1 + conflict
        term1 = self.__YL * term1  

        preds_unlabeled_sq_exp_f = np.exp(
            -self.__s * (preds_unlabeled * preds_unlabeled)
            )
        preds_unlabeled_sq_exp_f = preds_unlabeled_sq_exp_f * preds_unlabeled  

        g1 = (-1.0 / self.__size_l) * (self.__KLR.T @ term1)  
        g2 = ((-2.0 * self.__s * self.__lamU) / float(self.__size_u)
              ) * (self.__KUR.T @ preds_unlabeled_sq_exp_f)  
        g3 = 2.0 * self.__lam * (self.__KRR @ c_new) 

        grad = (g1 + g2 + g3).reshape(-1)
        return grad



# Sparse implementation 

class QN_S3VM_Sparse:
    """
    L-BFGS optimizer for S3VM with sparse data (SciPy sparse matrices).
    """

    parameters = {
        "lam": 1.0,
        "lamU": 1.0,
        "estimate_r": None,
        "minimum_labeled_patterns_for_estimate_r": 0,
        "BFGS_m": 50,
        "BFGS_maxfun": 500,
        "BFGS_factr": 1e12,
        "BFGS_pgtol": 1.0000000000000001e-05,
        "BFGS_verbose": -1,
        "surrogate_s": 3.0,
        "surrogate_gamma": 20.0,
        "breakpoint_for_exp": 500.0,
    }

    def __init__(self, X_l, L_l, X_u, random_generator, **kw):
        """
        Intializes the S3VM optimizer.
        """
    
        if random_generator is None:
            self.__random_generator = np.random.default_rng()
        elif isinstance(random_generator, (int, np.integer)):
            self.__random_generator = np.random.default_rng(
                int(random_generator))
        elif isinstance(random_generator, np.random.Generator):
            self.__random_generator = random_generator
        else:
            self.__random_generator = np.random.default_rng()

        # pad dims if needed
        if X_l.shape[1] > X_u.shape[1]:
            X_u = sparse.hstack(
                [X_u, sparse.coo_matrix(
                    (X_u.shape[0], X_l.shape[1] - X_u.shape[1]))
                    ])
            # sparse.coo_matrix creates a sparse matrix filled with zeros 
            # (in the COO = Coordinate Format format)
        elif X_l.shape[1] < X_u.shape[1]:
            X_l = sparse.hstack(
                [X_l, sparse.coo_matrix(
                    (X_l.shape[0], X_u.shape[1] - X_l.shape[1]))
                    ])

        # vertically stack the data matrices into one big matrix
        X = sparse.vstack([X_l, X_u])

        self.__size_l = int(X_l.shape[0])
        self.__size_u = int(X_u.shape[0])
        self.__size_n = self.__size_l + self.__size_u

        self.__YL = _col([int(l) for l in L_l], dtype=np.float64) 
        self.__setParameters(**kw)
        self.__kw = kw

        self.X_l = X_l.tocsr()
        self.X_u = X_u.tocsr()
        self.X = X.tocsr()
        
        # compute mean of unlabeled patterns
        self.__mean_u = np.asarray(self.X_u.mean(axis=0))  

        self.X_u_T = X_u.tocsc().T
        self.X_l_T = X_l.tocsc().T
        self.X_T = X.tocsc().T

        self.__needed_function_calls = 0
        self.__c = None

    def train(self):
        """
        Training phase.

        : return: list of predicted labels for the full training set
        """
        indi_opt = self.__optimize()
        self.__recomputeModel(indi_opt)
        predictions = self.getPredictions(self.X)
        return predictions

    def getPredictions(self, X, real_valued = False):
        """
        Computes the predicted labels for a given set of input data

        :arg(X, real_valued):
            X = input data matrix (sparse)
            real_valued = if True, then the real prediction values are returned;
                          if False, returns class labels

        : return: the predictions for X 
        """
        c_new = self.__c[:-1]  # (n,1)
        W = (self.X_T @ c_new) - (self.__mean_u.T * float(np.sum(c_new)))
        
        # Possibility of dimension mismatch due to use of sparse matrices
        if X.shape[1] > W.shape[0]:
            X = X[:, : W.shape[0]]
        if X.shape[1] < W.shape[0]:
            W = W[: X.shape[1], :]
        X = X.tocsc()
        preds = (X @ W) + float(self.__b)
        preds = np.asarray(preds).reshape(-1)
        if real_valued:
            return preds.tolist()
        return np.sign(np.sign(preds) + 0.1).astype(int).tolist()

    def predict(self, x):
        """
        Predicts a class label for a single input sample.

        :arg(x):
            x: input sample (array-like or sparse row)
            
        : return: predicted label in {-1, +1}
        """
        return self.getPredictions(sparse.csr_matrix(x), real_valued=False)[0]

    def predictValue(self, x):
        """
        Computes the real-valued decision function f(x) for 
        a single input sample (Representer Theorem).
    
        :arg(x):
            x = input sample (array-like or sparse row)
        
        : return: real-valued decision score f(x)
        """
        return self.getPredictions(sparse.csr_matrix(x), real_valued=True)[0]

    def getNeededFunctionCalls(self):
        """
        Returns the number of objective function evaluations 
        performed during the optimization process.
        
        : return: number of objective function calls (integer)
        """
        return int(self.__needed_function_calls)

    # Internal 

    def __setParameters(self, **kw):
        """
        Sets the model hyperparameters by merging user-provided values
        with the default configuration and storing the final settings 
        into internal variables.
    
        :arg(**kw):
            kw = dictionary of user-defined parameter overrides 
        
        : return: None
        """
        self.parameters = dict(self.parameters)
        self.parameters.update(kw)

        self.__lam = float(self.parameters["lam"])
        assert self.__lam > 0
        self.__lamU = float(self.parameters["lamU"])
        assert self.__lamU > 0
        self.__lam_Uvec = [self.__lamU * i for i in [
            0, 0.000001, 0.0001, 0.01, 0.1, 0.5, 1.0]]

        self.__minimum_labeled_patterns_for_estimate_r = float(
            self.parameters["minimum_labeled_patterns_for_estimate_r"])
        
        if self.parameters["estimate_r"] is not None:
            self.__estimate_r = float(self.parameters["estimate_r"])
        elif self.__YL.shape[0] >= self.__minimum_labeled_patterns_for_estimate_r:
            self.__estimate_r = float(np.mean(self.__YL))
        else:
            self.__estimate_r = 0.0

        self.__dim = self.__size_n + 1
        self.__BFGS_m = int(self.parameters["BFGS_m"])
        self.__BFGS_maxfun = int(self.parameters["BFGS_maxfun"])
        self.__BFGS_factr = float(self.parameters["BFGS_factr"])
       
        if sys.maxsize > 2**32:
            self.__BFGS_factr = 0.000488288 * self.__BFGS_factr
        self.__BFGS_pgtol = float(self.parameters["BFGS_pgtol"])
        self.__BFGS_verbose = int(self.parameters["BFGS_verbose"])
        self.__surrogate_gamma = float(self.parameters["surrogate_gamma"])
        self.__s = float(self.parameters["surrogate_s"])
        self.__breakpoint_for_exp = float(self.parameters["breakpoint_for_exp"])
        self.__b = float(self.__estimate_r)

    def __optimize(self):      
        """
        Runs the full optimization procedure.
    
        : return: (c_current, f_opt), where c_current contains the optimized 
                 coefficients and bias term, 
                 and f_opt is the final objective function value
        """
        self.__needed_function_calls = 0
        
        # starting_point
        c_current = np.zeros(self.__dim, dtype=np.float64)
        c_current[-1] = self.__b
        
        # Annealing sequence.
        for lamU in self.__lam_Uvec:
            self.__lamU = float(lamU)
            # crop one dimension (in case the offset b is fixed)
            c_current = c_current[:-1] 
            c_current = self.__localSearch(c_current)
            c_current = np.append(c_current, self.__b) 

        f_opt = self.__getFitness(c_current)
        return c_current, f_opt

    def __localSearch(self, start):     
        """
        Performs a local optimization step using the L-BFGS algorithm, 
        optimizing the coefficient vector while keeping the bias term fixed.
    
        :arg(start):
            start = initial coefficient vector (bias excluded)
        
        : return: optimized coefficient vector (bias excluded)
        """

        c_opt, f_opt, d = optimize.fmin_l_bfgs_b(
            self.__getFitness,
            start,
            m=self.__BFGS_m,
            fprime=self.__getFitness_Prime,
            maxfun=self.__BFGS_maxfun,
            factr=self.__BFGS_factr,
            pgtol=self.__BFGS_pgtol,
            iprint=self.__BFGS_verbose,
        )
        self.__needed_function_calls += int(d.get("funcalls", 0))
        return np.asarray(c_opt, dtype=np.float64)

    def __recomputeModel(self, indi):       
        """
        Stores the optimized parameter vector returned by the optimizer 
        into the internal model representation.
    
        :arg(indi):
            indi = optimization output whose first element contains 
                   the optimized parameter vector
        """
        self.__c = _col(indi[0], dtype=np.float64)

    def __getFitness(self, c):       
        """
        Evaluates the surrogate objective function for the sparse
        S3VM formulation.
    
        :arg(c):
            c = parameter vector 
        
        : return: scalar value of the surrogate objective function
        """

        c = np.asarray(c, dtype=np.float64).reshape(-1)
        if c.shape[0] == self.__dim - 1:
            c = np.append(c, self.__b)

        b = float(c[-1])
        c_new = c[:-1].reshape(-1, 1)
        c_new_sum = float(np.sum(c_new))
        
        # S3VM model predictions in centered feature space, 
        # using the optimized coefficients.
        # It is an alternative form of f(x) = < Phi(x) - mu_U, omega >,
        # where mu_U is the average of the unlabeled features.
        # Kernel representation: self.X_T*c_new = < Phi(x), omega >
        
        XTc = (self.X_T @ c_new) - (self.__mean_u.T * c_new_sum)
        
        # gamma * ( 1 - y_i f(x^i))
        preds_labeled = self.__surrogate_gamma * (
            1.0 - (self.__YL * (
                (self.X_l @ XTc) - (self.__mean_u @ XTc) + b)
                ))
        preds_unlabeled = (self.X_u @ XTc) - (self.__mean_u @ XTc) + b

        conflict = np.sign(
            np.sign(preds_labeled / self.__breakpoint_for_exp - 1.0
                    ) + 1.0)
        good = -1.0 * (conflict - 1.0)
        preds_labeled_for_conflicts = conflict * preds_labeled
        preds_labeled = preds_labeled * good
        
        # Compute values for good entries
        preds_labeled_log_exp = np.log(1.0 + np.exp(preds_labeled))
        # Compute values for instable entries
        preds_labeled_log_exp = good * preds_labeled_log_exp
        preds_labeled_final = preds_labeled_log_exp + preds_labeled_for_conflicts
        term1 = (
            1.0 / (self.__surrogate_gamma * self.__size_l)
            ) * float(np.sum(preds_labeled_final))

        preds_unlabeled_squared = preds_unlabeled * preds_unlabeled
        term2 = (
            float(self.__lamU) / float(self.__size_u)
            ) * float(np.sum(np.exp(-self.__s * preds_unlabeled_squared)))

        term3 = float(self.__lam * (
            c_new.T @ (self.X @ XTc - self.__mean_u @ XTc)))
        return term1 + term2 + term3

    def __getFitness_Prime(self, c):       
        """
        Computes the gradient of the surrogate objective function.
    
        :arg(c):
            c = parameter vector 
            
        : return: gradient vector with respect to the coefficients (bias excluded)
        """
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        if c.shape[0] == self.__dim - 1:
            c = np.append(c, self.__b)

        b = float(c[-1])
        c_new = c[:-1].reshape(-1, 1)
        c_new_sum = float(np.sum(c_new))

        XTc = (self.X_T @ c_new) - (self.__mean_u.T * c_new_sum)

        preds_labeled = self.__surrogate_gamma * (
            1.0 - (self.__YL * (
                (self.X_l @ XTc) - (self.__mean_u @ XTc) + b)))
        preds_unlabeled = (self.X_u @ XTc) - (self.__mean_u @ XTc) + b

        conflict = np.sign(np.sign(
            preds_labeled / self.__breakpoint_for_exp - 1.0) + 1.0)
        good = -1.0 * (conflict - 1.0)

        preds_labeled = preds_labeled * good
        preds_labeled_exp = np.exp(preds_labeled)

        term1 = preds_labeled_exp * (1.0 / (1.0 + preds_labeled_exp))
        term1 = good * term1
        term1 = term1 + conflict
        term1 = self.__YL * term1

        preds_unlabeled_sq_exp_f = np.exp(
            -self.__s * (preds_unlabeled * preds_unlabeled))
        preds_unlabeled_sq_exp_f = preds_unlabeled_sq_exp_f * preds_unlabeled

        term1_sum = float(np.sum(term1))
        tmp = (self.X_l_T @ term1) - (self.__mean_u.T * term1_sum)  

        g1 = (-1.0 / self.__size_l) * ((self.X @ tmp) - (self.__mean_u @ tmp))

        preds_unlabeled_sq_exp_f_sum = float(
            np.sum(preds_unlabeled_sq_exp_f))
        tmp_u = (self.X_u_T @ preds_unlabeled_sq_exp_f
                 ) - (self.__mean_u.T * preds_unlabeled_sq_exp_f_sum)

        g2 = (
            (-2.0 * self.__s * self.__lamU) / float(self.__size_u)
            ) * ((self.X @ tmp_u) - (self.__mean_u @ tmp_u))

        XTc_sum = float(np.sum(XTc))
        g3 = 2.0 * self.__lam * (
            (self.X @ XTc) - (self.__mean_u @ (XTc_sum)))

        grad = (g1 + g2 + g3).reshape(-1)
        return grad
