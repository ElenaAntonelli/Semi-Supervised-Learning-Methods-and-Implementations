# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 10:35:51 2026

@author: Elena Antonelli

Script to run LapSVM / LapTwinSVM / QN-S3VM on UCI datasets.
"""

# Import libraries
import os
import time
import random
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.datasets import fetch_openml

import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# EXPERIMENTAL SETUP

# Method: 'lapsvm' | 'laptwinsvm' | 's3vm'
METHOD = 's3vm'

# Dataset selector: specify either a string alias defined in OPENML_IDS 
#                   or an OpenML dataset ID (integer) or name
DATASET = "balance_scale"

# UCI datasets
OPENML_IDS = {
    "wine": 187,                     # Wine Recognition Data
    "breast_cancer": 15,             # Breast Cancer Wisconsin (Diagnostic)                 
    "pima_indians_diabetes": 37,     # Pima Indians Diabetes Database
    "seeds": 1499,                   # Seeds Dataset
    "ionosphere": 59,                # Ionosphere Dataset
    "balance_scale": 11,             # Balance Scale Dataset                 
    "haberman": 43,                  # Haberman's Survival Dataset
    "heart_disease": 53,             # Heart Disease (Cleveland)
    "german_credit": 31,             # German Credit Data (Statlog)
    "australian_credit": 40509       # Australian Credit Approval
}

# Class grouping for binary conversion
KEEP_CLASSES = {"neg": ["L","R"], "pos": ["B"]}
#KEEP_CLASSES = None

# Reproducibility
SEED = 42

# Repetitions and Optuna trials
N_RUNS = 5
N_TRIALS = 50 # number of different hyperparameter combinations

# Splits
TEST_SIZE = 0.40
VAL_SIZE_LABELED = 0.20
# VAL_SIZE_LABELED controls how much of the labeled data is used for validation 
# when looking for hyperparameters (Optuna)

# Fixed labeled fraction for ALL methods (for fair comparison)
LABELED_FRACS = [0.10, 0.20, 0.30, 0.40]

# Save policy
SAVE_WITH_TIMESTAMP = True



# OUTPUT UTILITIES 

def _base_dir():
    """
    Function that returns the base directory where result files are saved.
    
    :return: path to the base directory (string)       
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd() # getcwd = get the current working directory


def save_results_csv(rows, out_path, extra_summary_row=None): 
    """
    Function that saves run-level results and an optional summary row into a
    single CSV file.
    
    :arg(rows, out_path, extra_summary_row):
        rows = list of dictionaries, one dictionary per run 
        out_path = output file path where the CSV will be saved 
        extra_summary_row = optional dictionary appended as final row
    """
    if not rows:
        return

    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k) # keys = union of all possible columns
    
    # Add the summary keys (if any)
    if extra_summary_row is not None:
        for k in extra_summary_row.keys():
            if k not in keys:
                keys.append(k)

    df_runs = pd.DataFrame(rows) # converts results to pandas DataFrame
                                 # rows = runs
                                 # columns = keys
    # Append the summary row
    if extra_summary_row is not None:
        df_sum = pd.DataFrame([extra_summary_row])
        df_all = pd.concat([df_runs, df_sum], ignore_index=True, sort=False)
    else:
        df_all = df_runs

    df_all = df_all.reindex(columns=keys) # CSV with headers in the expected order
    
    # Automatically fix integer-like float columns (avoid .0) 
    for col in df_all.columns:
        if pd.api.types.is_float_dtype(df_all[col]): # if the column is type float
            series_no_na = df_all[col].dropna() # removes NaN values ​​temporarily
            if not series_no_na.empty and np.allclose(series_no_na, np.round(series_no_na)):
                df_all[col] = df_all[col].astype("Int64")


    # Write CSV (semicolon-separated), blank for NaNs/None, UTF-8
    df_all.to_csv(out_path, sep=';', index=False, encoding='utf-8', na_rep='', float_format="%.2f")
    # na_rep = '' -> NaN become empty cells


# DATASET LOADER

def load_openml_dataset(name_or_id):
    """
    This function loads a dataset from OpenML and returns the feature matrix X and labels y.
    It also handles basic preprocessing: missing value imputation,
    one-hot encoding for categorical features and conversion to NumPy arrays
    
    :arg(name_or_id):
        name_or_id = dataset identifier. It can be:
                    - a string alias defined in OPENML_IDS 
                    - an OpenML dataset ID (integer)
                    - an OpenML dataset name (string)
        
    :return:
        X = feature matrix; dataset 
        y = target labels        
    """
    if isinstance(name_or_id, str):
        key = name_or_id.strip().lower()
        # strip() removes spaces before/after and lower() makes lowercase
        if key in OPENML_IDS:
            name_or_id = OPENML_IDS[key] # dataset ID
    
    # Load dataset from OpenML (by ID or by name)
    if isinstance(name_or_id, (int, np.integer)):
        ds = fetch_openml(data_id=int(name_or_id), as_frame=True)
        # with as_frame=True ds.data becomes a pandas DataFrame
    else:
        ds = fetch_openml(name=str(name_or_id), as_frame=True)

    # dataset info
    ds.data.info()
    print("Unique labels:", ds.target.unique())
    print("Class counts:\n", ds.target.value_counts())

    X_df = ds.data.copy()
    y = np.asarray(ds.target)
    
    
    # If the classes are numbers but stored as strings, try to convert them to int
    if y.dtype.kind in ("U", "S", "O"):
        # U = unicode string
        # S = byte string
        # O = object
        try:
            y_num = y.astype(np.int64)
            y = y_num
        except Exception:
            pass
        
    # Identify numeric vs categorical columns 
    num_cols = X_df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    # Impute missing values
    # Numeric: fill with median
    if len(num_cols) > 0:
        for c in num_cols:
            # Try to convert each value in column c into a number
            # If a value is not convertible, replace it with NaN
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")  
            X_df[c] = X_df[c].fillna(X_df[c].median())

    # Categorical: fill with most frequent (mode)
    if len(cat_cols) > 0:
        for c in cat_cols:           
            X_df[c] = X_df[c].astype("object") # treat each column as object
            mode = X_df[c].mode(dropna=True) # dropna=True ignores NaNs
            fill_value = mode.iloc[0] if len(mode) > 0 else "missing"
            X_df[c] = X_df[c].fillna(fill_value)

        # One-hot encode categoricals 
        X_df = pd.get_dummies(X_df, columns=cat_cols, dummy_na=False)
       

    # Return numpy arrays (float features) 
    X = X_df.to_numpy(dtype=np.float64)
    return X, y


def _resolve_keep_classes(y, keep_classes):
    """
    Function that validates and resolves the class grouping specified in KEEP_CLASSES.
      
    :arg(y, keep_classes):
        y = array of original labels
        keep_classes = dictionary defining the grouping of original labels into
                       negative and positive classes:
                       {"neg": [...], "pos": [...]}
        
    :return:
        (neg_group, pos_group) = tuple of labels assigned to the negative class
                                 and tuple of labels assigned to the positive class
    """
    # Each element in neg/pos must be a real label (present in np.unique(y))
       
    u = list(np.unique(y)) # find all different labels in y

    if keep_classes is None:
        return None

    if not isinstance(keep_classes, dict) or "neg" not in keep_classes or "pos" not in keep_classes:
        raise ValueError("KEEP_CLASSES must be a dict with keys {'neg','pos'}.")

    neg_items = list(keep_classes["neg"])
    pos_items = list(keep_classes["pos"])

    if len(neg_items) == 0 or len(pos_items) == 0:
        raise ValueError("'neg' and 'pos' must contain at least one class.")

    for v in neg_items + pos_items:
        if v not in u:
            raise ValueError(
                f"Value {v} in KEEP_CLASSES is not a valid label. "
                f"Valid labels are: {sorted(u)}"
            )

    if set(neg_items) & set(pos_items):
        raise ValueError(
            "KEEP_CLASSES groups overlap: same class appears in both 'neg' and 'pos'."
        )

    return (tuple(neg_items), tuple(pos_items))


def make_binary(X, y, keep_classes=None):
    """
    This function prepares a dataset for binary semi-supervised classification methods.
    It optionally selects/aggregates classes to obtain a binary problem and
    remaps labels to {-1, +1}.
    
    :arg(X, y, keep_classes):
        X = feature matrix; dataset
        y = original labels
        keep_classes = optional class grouping to force a binary setting, dict or None:
                       {"neg": [...], "pos": [...]}
                       If None, the dataset must already contain exactly 2 classes.
        
    :return:
        X = feature matrix
        y_pm = remapped binary labels in {-1, +1}
        (neg_group, pos_group) = tuples containing the original labels mapped to -1 and +1
        
    """  
    y = np.asarray(y).reshape(-1)
    X = np.asarray(X)

    u = list(np.unique(y))
    
    if keep_classes is None:
        if len(u) != 2:
            raise ValueError(
                f"Dataset is not binary (found {len(u)} classes). "
                f"Define KEEP_CLASSES {{'neg': [...], 'pos': [...]}}. Labels: {u}")       
        selected = (u[0], u[1])
    else:
        selected = _resolve_keep_classes(y, keep_classes)
    
    neg_group, pos_group = selected  # extract the two groups of classes
    
    # normalize scalars (especially strings) to 1-element lists
    if isinstance(neg_group, (str, bytes)) or not hasattr(neg_group, "__iter__"):
        neg_group = [neg_group]
    if isinstance(pos_group, (str, bytes)) or not hasattr(pos_group, "__iter__"):        
        pos_group = [pos_group]
        
    neg_set = set(neg_group)
    pos_set = set(pos_group)
    
    if neg_set.intersection(pos_set):
        raise ValueError(f"KEEP_CLASSES groups overlap: {sorted(list(neg_set & pos_set))}")

    
    allowed = list(neg_set | pos_set) # union
    mask = np.isin(y, allowed)
    # mask is a boolean array that keeps only samples whose labels are allowed
    # It is used to discard any classes that are not considered in the binary setting

    X = X[mask]
    y = y[mask]
    
    # map: neg -> -1, pos -> +1
    y_pm = np.where(np.isin(y, list(pos_set)), +1, -1)

    return X, y_pm, (neg_group, pos_group)


def train_test_from_dataset(seed=42, test_size=0.4):
    """
    Function that loads the selected OpenML dataset, converts it to a binary problem,
    splits the data into training and test sets using a fixed random seed
    and standardizes features.
    
    :arg(seed, test_size):
        seed = random seed used for reproducibility (int)
        test_size = fraction of samples assigned to the test set (float in (0,1))
        
    :return:
        rng = NumPy random generator initialized with seed (Generator)
        X_train = training features, ndarray shape (n_train, n_features)
        X_test = test features, ndarray shape (n_test, n_features)
        y_train = training labels in {-1, +1}, ndarray shape (n_train, )
        y_test = test labels in {-1, +1}, ndarray shape (n_test, )
        kept = (neg_group, pos_group), tuples of original labels mapped to -1 and +1
        
    """  
    rng = default_rng(seed)
    X, y = load_openml_dataset(DATASET)
    X, y, kept = make_binary(X, y, keep_classes=KEEP_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Scale to mean = 0 and variance = 1
    # Standardization is fitted on the training set only and then applied to the test set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return rng, X_train, X_test, y_train, y_test, kept


def compute_metrics_percent(y_true, y_pred):
    """
    Compute classification metrics in percentage form.

    :arg(y_true, y_pred):
        y_true = true binary labels in {-1, +1}
        y_pred = predicted labels

    :return:
        acc = accuracy in percentage
        prec = precision in percentage
        rec = recall in percentage
        f1 = F1-score in percentage
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel()

    if not np.all(np.isin(np.unique(y_pred), [-1, 1])):
        y_pred = np.where(y_pred >= 0, 1, -1)

    y_pred = y_pred.astype(int)

    acc = 100.0 * accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="binary",
        pos_label=1,
        zero_division=0
    )
    return round(acc, 2), round(100*p, 2), round(100*r, 2), round(100*f1, 2)


# MODEL SELECTION

def _import_method(method):
    """
    Function that selects and imports the requested SSL method based on a string identifier.
    It returns a dictionary containing the model class and, when available, the kernel function.
    
    :arg(method):
        method = method identifier string. Allowed values are:
                 'lapsvm', 'laptwinsvm', 's3vm'
        
    :return: imports = dictionary with the following keys:
                      - 'name'  : method name (string)
                      - 'Model' : model class to instantiate
                      - 'rbf'   : RBF kernel function (for LapSVM/LapTwinSVM) or None (for S3VM)        
    """
    m = method.lower().strip()
    
    if m == 'lapsvm':
        from LapSVM_method import LapSVM, rbf
        return {'name': 'lapsvm', 'Model': LapSVM, 'rbf': rbf}
    if m == 'laptwinsvm':
        from LapTwinSVM_method import LapTwinSVM, rbf
        return {'name': 'laptwinsvm', 'Model': LapTwinSVM, 'rbf': rbf}
    if m == 's3vm':
        from S3VM_method import QN_S3VM
        return {'name': 's3vm', 'Model': QN_S3VM, 'rbf': None}
    
    raise ValueError("METHOD must be: 'lapsvm', 'laptwinsvm', or 's3vm'.")


# OPTUNA OBJECTIVES 

def make_objective(method, imports, X_l, y_l, X_u, seed=42, val_size=0.3):
    """
    Function that builds and returns an Optuna objective function for
    hyperparameter tuning. The returned objective trains the selected method
    on a subset of labeled data and evaluates its accuracy on an internal validation split.
    
    :arg(method, imports, X_l, y_l, X_u, seed, val_size):
        method = method identifier string: 'lapsvm', 'laptwinsvm', or 's3vm'
        imports = dictionary containing the model class and (optionally) the kernel
        X_l = labeled feature matrix
        y_l = labeled targets in {-1, +1}
        X_u = unlabeled feature matrix
        seed = random seed used for the internal train/validation split (int)
        val_size = fraction of labeled data used as validation within each trial (float in (0,1))
        
    :return: Optuna objective function that takes a trial and returns a 
             validation accuracy        
    """
    method = method.lower().strip()

    if method in ('lapsvm', 'laptwinsvm'):
        Model = imports['Model']
        rbf = imports['rbf']

        def objective(trial):
            # hyperparameters to test:
            neighbor_mode = trial.suggest_categorical('neighbor_mode', ['distance', 'connectivity'])
            gamma_A = trial.suggest_float('gamma_A', 1e-4, 10.0, log=True)
            gamma_I = trial.suggest_float('gamma_I', 1e-4, 10.0, log=True)
            t = trial.suggest_float('t', 1e-3, 10.0, log=True)
            n_neighbor = trial.suggest_int('n_neighbor', 3, 15)
            gamma_kernel = trial.suggest_float('gamma_kernel', 1e-3, 10.0, log=True)
            
            # Internal splitting of labeled data between train and validation sets
            X_l_train, X_l_val, y_l_train, y_l_val = train_test_split(
                X_l, y_l, test_size=val_size, random_state=seed, stratify=y_l
            )
            
            # Parameters
            opt = {
                'neighbor_mode': neighbor_mode,
                'n_neighbor': n_neighbor,
                't': t,
                'kernel_function': rbf,
                'kernel_parameters': {'gamma': gamma_kernel},
                'gamma_A': gamma_A,
                'gamma_I': gamma_I,
            }
            
            # Training and evaluation
            try:
                model = Model(opt)
                model.fit(X_l_train, y_l_train, X_u)
                acc = float(model.accuracy(X_l_val, y_l_val))
                return acc           
            except np.linalg.LinAlgError:
                # singular matrix or numerical issue
                raise optuna.TrialPruned()
            except Exception as e:
                print('Trial error:', e)
                raise optuna.TrialPruned()

        return objective

    if method == 's3vm':
        QN_S3VM = imports['Model']

        def objective(trial):
            # hyperparameters to test:
            lam = trial.suggest_float('lam', 1e-5, 10.0)
            lamU = trial.suggest_float('lamU', 1e-5, 10.0)
            kernel_type = trial.suggest_categorical('kernel_type', ['Linear', 'RBF'])
            sigma = trial.suggest_float('sigma', 1e-2, 10.0) if kernel_type == 'RBF' else 1.0
            n_total = len(X_l) + len(X_u)
            numR = trial.suggest_int('numR', max(10, int(0.2 * n_total)), n_total)
            
            # Internal splitting of labeled data between train and validation sets
            X_l_train, X_l_val, y_l_train, y_l_val = train_test_split(
                X_l, y_l, test_size=val_size, random_state=seed, stratify=y_l
            )

            rg = random.Random(seed)
            
            # Training and evaluation
            try:
                model = QN_S3VM(
                    X_l_train,
                    y_l_train.tolist(),
                    X_u,
                    rg,
                    lam=lam,
                    lamU=lamU,
                    kernel_type=kernel_type,
                    sigma=sigma,
                    numR=numR,
                )
                model.train()
                return float(model.accuracy(X_l_val, y_l_val))
            except Exception as e:
                print('Trial error:', e)
                return 0.0

        return objective

    raise ValueError(f"Unknown method: {method}")


# FINAL TRAINING

def train_final_and_eval(method, imports, best_params, X_l, y_l, X_u, X_test, y_test, seed=42):
    """
    This function trains the selected method using the best hyperparameters 
    found by Optuna and evaluates the trained model on the test set.
      
    :arg(method, imports, best_params, X_l, y_l, X_u, X_test, y_test, seed):
        method = method identifier string: 'lapsvm', 'laptwinsvm', or 's3vm'
        imports = dictionary containing the model class and (optionally) the kernel
        best_params = dictionary containing the best hyperparameters found by Optuna (dict)
        X_l = labeled feature matrix
        y_l = labeled targets in {-1, +1}
        X_u = unlabeled feature matrix
        X_test = test feature matrix
        y_test = test targets in {-1, +1}
        seed = random seed used for reproducibility when required 
        
    :return:
        train_acc = accuracy on the labeled training set 
        test_acc = accuracy on the test set 
        test_prec = precision on the test set 
        test_rec = recall on the test set 
        test_f1 = F1-score on the test set 
        pred_counts = dictionary with counts of predicted labels on the test set        
    """
    method = method.lower().strip()

    if method in ('lapsvm', 'laptwinsvm'):
        Model = imports['Model']
        rbf = imports['rbf']
        
        # Train the model with the best hyperparameters and evaluate on the test set
        opt = {
            'neighbor_mode': best_params['neighbor_mode'],
            'n_neighbor': best_params['n_neighbor'],
            't': best_params['t'],
            'kernel_function': rbf,
            'kernel_parameters': {'gamma': best_params['gamma_kernel']},
            'gamma_A': best_params['gamma_A'],
            'gamma_I': best_params['gamma_I'],
        }

        model = Model(opt)
        model.fit(X_l, y_l, X_u)

        train_acc = float(model.accuracy(X_l, y_l))
        
        # Prediction and evaluation on test set
        y_pred_test = model.predict(X_test)
        test_acc = float(model.accuracy(X_test, y_test))
        _, test_prec, test_rec, test_f1 = compute_metrics_percent(y_test, y_pred_test)
       
        pred_counts = {
            'pred_-1_test': int(np.sum(np.asarray(y_pred_test) == -1)),
            'pred_+1_test': int(np.sum(np.asarray(y_pred_test) == 1)),
        }
        return train_acc, test_acc, test_prec, test_rec, test_f1, pred_counts

    if method == 's3vm':
        QN_S3VM = imports['Model']
        rg_final = random.Random(seed)
        sigma = best_params.get('sigma', 1.0) 
        # sigma is only present when kernel_type = 'RBF'; use default otherwise

        model = QN_S3VM(
            X_l,
            y_l.tolist(),
            X_u,
            rg_final,
            lam=best_params['lam'],
            lamU=best_params['lamU'],
            kernel_type=best_params['kernel_type'],
            sigma=sigma,
            numR=best_params['numR'],
        )
        model.train()

        train_acc = float(model.accuracy(X_l, y_l))
        y_pred_test = model.getPredictions(X_test, real_valued=False)
        test_acc = float(model.accuracy(X_test, y_test))
        _, test_prec, test_rec, test_f1 = compute_metrics_percent(y_test, y_pred_test)
        
        pred_counts = {
            'pred_-1_test': int(np.sum(np.asarray(y_pred_test) == -1)),
            'pred_+1_test': int(np.sum(np.asarray(y_pred_test) == 1)),
        }
        return train_acc, test_acc, test_prec, test_rec, test_f1, pred_counts

    raise ValueError(f"Unknown method: {method}")


# MAIN

def main():
    method = METHOD.lower().strip()
    imports = _import_method(method)

    # Keep the global train/test split fixed for comparability across runs.
    # Variability across runs is introduced by changing the labeled/unlabeled 
    # split using a different seed per run.
    _, X_train, X_test, y_train, y_test, kept_classes = train_test_from_dataset(
        seed=SEED, test_size=TEST_SIZE
    )
    
    rows = []
    
    for labeled_frac in LABELED_FRACS:
        n_labeled = max(2, int(labeled_frac * len(X_train))) # at least 2 labeled
  
        test_accs_frac = []
        test_precs_frac = []
        test_recs_frac = []
        test_f1s_frac = []

        for run_idx in range(1, N_RUNS + 1):
            run_seed = int(SEED + run_idx)
            print(f"=== frac={labeled_frac:.2f} RUN {run_idx}/{N_RUNS} ({method}) seed={run_seed} ===")

            # Different labeled/unlabeled split at each run
            rng_run = default_rng(run_seed)
            
            # Stratified labeled sampling: ensure both classes are present 
            classes = np.unique(y_train)
            if len(classes) != 2:
                raise ValueError(f"Expected binary labels in y_train, found {classes}")
            
            # minimum labeled per class 
            min_per_class = 2
            
            # if too few labeled, force feasibility
            if n_labeled < 2 * min_per_class:
                n_labeled = 2 * min_per_class
            
            idx_c0 = np.where(y_train == classes[0])[0]
            idx_c1 = np.where(y_train == classes[1])[0]
            
            # guard: training split must contain both classes
            if len(idx_c0) == 0 or len(idx_c1) == 0:
                raise ValueError("Training set contains only one class; cannot run binary SSL.")
            
            # sample min_per_class from each class
            labeled_c0 = rng_run.choice(idx_c0, size=min_per_class, replace=False)
            labeled_c1 = rng_run.choice(idx_c1, size=min_per_class, replace=False)
            
            labeled_idx = np.concatenate([labeled_c0, labeled_c1])
            
            # sample the remaining labeled points from the rest (no constraint)
            remaining = n_labeled - labeled_idx.size
            if remaining > 0:
                pool = np.setdiff1d(np.arange(len(X_train)), labeled_idx, assume_unique=False)
                extra = rng_run.choice(pool, size=remaining, replace=False)
                labeled_idx = np.concatenate([labeled_idx, extra])
            
            # shuffle to remove any ordering bias
            rng_run.shuffle(labeled_idx)
            
            # unlabeled 
            unlabeled_idx = np.setdiff1d(np.arange(len(X_train)), labeled_idx, assume_unique=False)

            X_l = X_train[labeled_idx]
            y_l = y_train[labeled_idx]
            X_u = X_train[unlabeled_idx]

            objective = make_objective(
                method, imports, X_l, y_l, X_u, seed=run_seed, val_size=VAL_SIZE_LABELED
            )
        
            # Create the Optuna Study
            study = optuna.create_study(direction='maximize') # maximize validation accuracy
            study.optimize(objective, n_trials=N_TRIALS)

            # Convert Optuna params to pure Python types (safe for CSV / printing)
            best_params_clean = {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in study.best_params.items() # k = parameter name (string)
                                                      # v = best value found
            }
        
            param_cols = {f'param_{k}': best_params_clean[k] for k in best_params_clean}
    
            print("Best params:", best_params_clean)
            print(f"Best val acc: {study.best_value:.2f}%")
            
            # Final training + test evaluation
            train_acc, test_acc, test_prec, test_rec, test_f1, pred_counts  = train_final_and_eval(
                method,
                imports,
                best_params_clean,
                X_l,
                y_l,
                X_u,
                X_test,
                y_test,
                seed=run_seed,
            )

            test_accs_frac.append(test_acc)
            test_precs_frac.append(test_prec)
            test_recs_frac.append(test_rec)
            test_f1s_frac.append(test_f1)
                    
            # Store run results 
            rows.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'method': method,
                'dataset': str(DATASET),
                'kept_neg': ",".join(map(str, kept_classes[0])),
                'kept_pos': ",".join(map(str, kept_classes[1])),
                'seed_global_split': int(SEED), # seed used for the fixed train/test split
                'seed_run': int(run_seed), # seed used for the labeled/unlabeled split in the current run
                'run': int(run_idx),
                'n_trials': int(N_TRIALS), # number of Optuna trials for hyperparameter optimization
                'test_size': float(TEST_SIZE),
                'val_size_labeled': float(VAL_SIZE_LABELED),
                'labeled_frac': float(labeled_frac),
                'n_train_total': int(len(X_train)),
                'n_labeled': int(len(X_l)),
                'n_unlabeled': int(len(X_u)),
                'val_acc_best': round(float(study.best_value), 2), # best validation accuracy found by Optuna
                'train_acc_labeled': round(float(train_acc), 2),
                'test_acc': round(float(test_acc), 2),
                'test_precision': test_prec,
                'test_recall': test_rec,
                'test_f1': test_f1,
                **pred_counts, # counts of predicted labels (-1 / +1) on the test set
                **param_cols, # best hyperparameters found by Optuna
            })

            print(f"Train (labeled) acc: {train_acc:.2f}%")
            print(f"Test acc: {test_acc:.2f}%")
            print(f"Test precision: {test_prec:.2f}%")
            print(f"Test recall: {test_rec:.2f}%")
            print(f"Test F1: {test_f1:.2f}%")
    
        # Compute summary statistics, and save everything to CSV
        mean_test = float(np.mean(test_accs_frac)) if test_accs_frac else 0.0
        std_test = float(np.std(test_accs_frac)) if test_accs_frac else 0.0
    
        rows.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'method': method,
                'dataset': str(DATASET),
                'run': 'SUMMARY',
                'labeled_frac': float(labeled_frac),
                'mean_test_acc': round(mean_test, 2),
                'std_test_acc': round(std_test, 2),
                'mean_test_precision': round(np.mean(test_precs_frac), 2),
                'std_test_precision': round(np.std(test_precs_frac), 2),
                
                'mean_test_recall': round(np.mean(test_recs_frac), 2),
                'std_test_recall': round(np.std(test_recs_frac), 2),
                
                'mean_test_f1': round(np.mean(test_f1s_frac), 2),
                'std_test_f1': round(np.std(test_f1s_frac), 2),
                'n_runs': int(N_RUNS),
            })

    base_dir = _base_dir()
    results_dir = os.path.join(base_dir, "Results_Classification")
    os.makedirs(results_dir, exist_ok=True)
    
    if SAVE_WITH_TIMESTAMP:
        fname = f"results_{DATASET}_{method}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        fname = f"results_{DATASET}_{method}.csv"
    
    out_path = os.path.join(results_dir, fname)
    save_results_csv(rows, out_path, extra_summary_row=None)


    print("\n=== DONE ===")
    print("Saved:", out_path)
    print(f"Mean test acc over {N_RUNS} runs: {mean_test:.2f}% (std {std_test:.2f}%)")


if __name__ == '__main__':
    main()
