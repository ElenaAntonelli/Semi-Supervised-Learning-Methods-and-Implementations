# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 10:20:31 2026

@author: Elena Antonelli


Script for semi-supervised constrained clustering analyses.

Supported methods:
  - SpectralSSC  
  - MPCKMeans 
"""

# Import libraries
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from numpy.random import default_rng

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

import matplotlib.pyplot as plt


# EXPERIMENTAL SETUP

# Dataset selector: OpenML id (int) or alias (str) in OPENML_IDS 
#                   or an OpenML dataset name (str)
DATASET = "ionosphere"    

# Method: "spectral" | "mpck" | "both"
METHOD = "spectral"

# UCI datasets
OPENML_IDS = {
    "iris": 61,                     # Iris Plants Dataset
    "wine": 187,                    # Wine Recognition Data
    "breast_cancer": 15,            # Breast Cancer Wisconsin (Diagnostic)
    "seeds": 1499,                  # Seeds Dataset
    "ionosphere": 59,               # Ionosphere Dataset
    "heart_disease": 53,            # Heart Disease (Cleveland) 
    "soybean": 42,                  # Soybean (Large) Dataset
    "glass": 41,                    # Glass Identification Dataset
    "ecoli": 39,                    # Ecoli Protein Localization Dataset   
    "zoo": 62                       # Zoo Dataset
}

# Number of repetitions for each constraint level 
N_RUNS = 5

# Fraction of points used to sample constraints 
LABELED_FRAC = 0.20

# Total constraints grid (ML+CL)
CONSTRAINTS_GRID = [50, 100, 200, 400]

# Split between ML and CL
ML_RATIO = 0.5  # 0.5 -> half ML, half CL


# Standardize features
STANDARDIZE = True

# Reproducibility
SEED = 123

# ----- SpectralSSC params -----
GRAPH_TYPE = "knn"              # "knn" or "rbf"
K_NN = 10                       # used if graph_type = "knn"
GAMMA = None                    # (default = None)
KMEANS_N_INIT = 100
NORMALIZED_LAPLACIAN = False    # True: normalized Laplacian; False: unnormalized
DELTA = 0.0                     # target minimum average constraint satisfaction (trace(V^T Q V)/k)
BETA_SEARCH = True              # enable 1D beta search
BETA_SEARCH_ITERS = 300
W_ML = 3.0                      # must-link weight in Q
W_CL = 1.0                      # cannot-link weight in Q


# ----- MPCKMeans params -----
MPCK_W = 1.0
MPCK_MAX_ITER = 150



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
            name_or_id = OPENML_IDS[key] #dataset ID
    
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
        # dummy_na : do not create a separate column for missing values

    # Return numpy arrays (float features) 
    X = X_df.to_numpy(dtype=np.float64)
    return X, y


def prepare_X_y(X, y, standardize):
    """
    This function prepares the input data matrix X and the label vector y.
    It maps the original labels to consecutive integers in {0, ..., C-1} and,
    if required, standardizes the features.

    :arg(X, y, standardize):
        X = feature matrix, array-like of shape (n_samples, n_features)
        y = target labels, array-like of shape (n_samples,)
        standardize = boolean flag. If True, X is standardized 

    :return:
        X = processed feature matrix
        y_int = integer-mapped labels in {0, ..., C-1}
    """
    y = np.asarray(y).reshape(-1)
    uniq = np.unique(y)
    y_int = np.zeros_like(y, dtype=int)
    
    # Label remapping
    for i, lab in enumerate(uniq): # enumerate(uniq) = pairs (index, value)
        y_int[y == lab] = i
    
    print("n_classes:", len(uniq), "mapped to:", np.unique(y_int))
   
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        X = StandardScaler().fit_transform(X)
    
    return X, y_int


# CONSTRAINT GENERATION

def clean_constraints(must_link, cannot_link):
    """
    This function cleans the input pairwise constraints by removing invalid pairs,
    enforcing a consistent ordering of indices in each pair and 
    resolving conflicts between must-link and cannot-link constraints.

    :arg(must_link, cannot_link):
        must_link = list of must-link constraints, where each element is a pair (i, j)
        cannot_link = list of cannot-link constraints, where each element is a pair (i, j)

    :return:
        must_link_clean = cleaned must-link constraints as a list of unique pairs (i, j) with i < j
        cannot_link_clean = cleaned cannot-link constraints as a list of unique pairs (i, j) with i < j
    """
    ml = set()
    for i, j in must_link:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        ml.add((a, b))
    
    cl = set()
    for i, j in cannot_link:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        cl.add((a, b)) 
    
    # sets automatically remove duplicate pairs
        
    inter = ml.intersection(cl)
    if inter:
        # if there are conflicts, it removes them from both sets
        ml.difference_update(inter)
        cl.difference_update(inter)
    
    return list(ml), list(cl)


def sample_constraints_class_balanced(y, labeled_idx, n_ml, n_cl, rng):
    """
    This function samples pairwise constraints (must-link and cannot-link) 
    from a labeled subset. Must-link constraints are generated in a class-balanced way, 
    so that different classes contribute more evenly to the sampled pairs.

    :arg(y, labeled_idx, n_ml, n_cl, rng):
        y = true labels
        labeled_idx = indices of the labeled subset used to generate constraints                
        n_ml = requested number of must-link constraints 
        n_cl = requested number of cannot-link constraints 
        rng = NumPy random generator 

    :return:
        must_link = list of sampled must-link pairs (i, j)
        cannot_link = list of sampled cannot-link pairs (i, j)
    """
    labeled_idx = np.asarray(labeled_idx, dtype=int)
    y_lab = y[labeled_idx]

    by_class = {}
    for idx, c in zip(labeled_idx, y_lab):
        by_class.setdefault(int(c), []).append(int(idx))

    classes = sorted(by_class.keys()) # ordered list of classes present in the labeled subset
    # consider only classes that have at least 2 points (for must-link constraints)
    valid_ml_classes = [c for c in classes if len(by_class[c]) >= 2]
    if len(classes) < 2:
        return [], []

    must_link = []
    cannot_link = []
    
    # Must-link
    if n_ml > 0 and len(valid_ml_classes) > 0:
        # approximate ML quota per valid class
        per = max(1, n_ml // max(1, len(valid_ml_classes)))
        
        for c in valid_ml_classes:
            pool = by_class[c] # list of indices (in the labeled subset) of class c
            k = min(per, n_ml - len(must_link)) # how many MLs to generate from class c
            for _ in range(k):
                i, j = rng.choice(pool, size=2, replace=False)
                must_link.append((int(i), int(j)))
                if len(must_link) >= n_ml:
                    break
            if len(must_link) >= n_ml:
                break
        
        # Final filling of the missing must-links
        while len(must_link) < n_ml:
            c = int(rng.choice(valid_ml_classes))
            pool = by_class[c]
            i, j = rng.choice(pool, size=2, replace=False)
            must_link.append((int(i), int(j)))
    
    # Cannot-link
    if n_cl > 0:
        for _ in range(n_cl):
            c1, c2 = rng.choice(classes, size=2, replace=False)
            i = int(rng.choice(by_class[int(c1)]))
            j = int(rng.choice(by_class[int(c2)]))
            cannot_link.append((i, j))

    return clean_constraints(must_link, cannot_link)


def constraint_violations(labels, must_link, cannot_link):
    """
    This function computes the number of violated must-link and cannot-link
    constraints given a clustering assignment.

    :arg(labels, must_link, cannot_link):
        labels = predicted cluster labels
        must_link = list of must-link pairs (i, j)
        cannot_link = list of cannot-link pairs (i, j)

    :return:
        ml_v = number of violated must-link constraints
        cl_v = number of violated cannot-link constraints
    """
    ml_v = 0
    for i, j in must_link:
        if i != j and labels[i] != labels[j]:
            ml_v += 1
    cl_v = 0
    for i, j in cannot_link:
        if i != j and labels[i] == labels[j]:
            cl_v += 1
    return ml_v, cl_v


# METRICS

def _comb2(x):
    """
    This function computes the number of unordered pairs that can be formed
    from each element of x, i.e. the binomial coefficient C(x, 2).

    :arg(x):
        x = integer or array-like of non-negative integers

    :return: number of distinct pairs 
    """
    x = np.asarray(x, dtype=np.int64)
    return x * (x - 1) // 2


def pairwise_scores(y_true, y_pred):
    """
    Function that computes pairwise clustering scores by comparing
    reference labels and predicted clustering at the pair level.

    :arg(y_true, y_pred):
        y_true = reference labels
        y_pred = predicted cluster labels

    :return:
        prec = pairwise precision
        rec  = pairwise recall
        f1   = pairwise F1-score
    """
    # TP: pairs that are in the same true class and the same predicted cluster
    # FP: pairs in the same predicted cluster but different true classes
    # FN: pairs in the same true class but different predicted clusters
 
    cm = contingency_matrix(y_true, y_pred, sparse=False) 
    # cm[a,b] = number of points that have true class a and have been assigned to cluster b
    
    tp = int(_comb2(cm).sum())

    true_sizes = cm.sum(axis=1) # real class sizes (sum across rows)
    pred_sizes = cm.sum(axis=0) # predicted cluster sizes (sum across columns)

    true_same = int(_comb2(true_sizes).sum())
    # true_same = total number of pairs that are in the same true class 
    pred_same = int(_comb2(pred_sizes).sum()) # total number of pairs that are in the same predicted cluster

    fp = pred_same - tp
    fn = true_same - tp

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0 #  harmonic mean
    return float(prec), float(rec), float(f1)


# METHODS

def run_spectral_ssc(X, K, must_link, cannot_link, params, seed):
    """
    Function that runs the SpectralSSC method using the given dataset and pairwise constraints.

    :arg(X, K, must_link, cannot_link, params, seed):
        X = feature matrix
        K = target number of clusters 
        must_link = list of must-link constraints (i, j)
        cannot_link = list of cannot-link constraints (i, j)
        params = dictionary containing SpectralSSC hyperparameters
        seed = random seed used for reproducibility 
        
    :return:
        labels = predicted cluster labels
        extras = dictionary with additional outputs 
    """
    from SpectralSSC_method import SpectralSSC
    
    # Validate categorical parameter graph_type
    gt = str(params.get("graph_type", "knn")).lower()
    if gt not in ("knn", "rbf"):
        raise ValueError(f"Unknown graph_type: {gt}")
    graph_type = gt

    model = SpectralSSC(
        n_clusters=int(K),
        graph_type=graph_type,
        n_neighbors=int(params.get("n_neighbors", 20)),
        gamma=params.get("gamma", None),
        normalized=bool(params.get("normalized", True)),
        delta=float(params.get("delta", 0.0)),
        beta_search=bool(params.get("beta_search", True)),
        beta_search_iters=int(params.get("beta_search_iters", 30)),
        kmeans_n_init=int(params.get("kmeans_n_init", 50)),
        random_state=int(seed)
    )

    labels = model.fit_predict(
        X,
        must_link=must_link,
        cannot_link=cannot_link,
        must_weight=float(params.get("w_ml", 1.0)),
        cannot_weight=float(params.get("w_cl", 1.0))
    )

    extras = {
        "beta": getattr(model, "beta_", None)
    }
    print("beta:", model.beta_)
   
    return np.asarray(labels, dtype=int), extras


def run_mpckmeans(X, K, must_link, cannot_link, params, seed):
    """
    This function runs the MPCKMeans algorithm using the given dataset and
    pairwise constraints.

    :arg(X, K, must_link, cannot_link, params, seed):
        X = feature matrix
        K = target number of clusters 
        must_link = list of must-link constraints (i, j)
        cannot_link = list of cannot-link constraints (i, j)
        params = dictionary containing MPCKMeans hyperparameters
        seed = random seed used for reproducibility 

    :return:
        labels = predicted cluster labels
        extras = dictionary with additional outputs 
    """  
    from MPCK_Means_method import MPCKMeans

    # Forward extra MPCK hyperparameters if present in params dict
    extra_kwargs = {k: v for k, v in params.items() if k not in ('w', 'max_iter')}
    
    model = MPCKMeans(
        K=int(K),
        w=float(params.get('w', 1.0)),
        max_iter=int(params.get('max_iter', 150)),
        random_state=int(seed),
        **extra_kwargs,
    )
    labels, mu, M, neighborhoods = model.fit(X, must_link=must_link, cannot_link=cannot_link)

    extras = {
        "n_neighborhoods": 0 if neighborhoods is None else int(len(neighborhoods)),
    }
    return np.asarray(labels, dtype=int), extras


# AGGREGATION

def aggregate(df):
    """
    This function aggregates run-level results by grouping them according to the
    method and the requested number of constraints, and computing mean and standard
    deviation for the main evaluation metrics.

    :arg(df):
        df = pandas DataFrame containing one row per run, with metric columns

    :return: aggregated DataFrame containing mean and std for each metric
             at each constraint level (grouped by method and constraint settings)
    """
    group_cols = ["method", "n_constraints_total", "n_ml_req", "n_cl_req"]
    metrics = [
        "nmi", "cri", "pair_prec", "pair_rec", "pair_f1",
        "ml_sat", "cl_sat", "time_sec"
    ]
    # group rows by method and constraint setting, compute mean and std 
    # of metrics, and restore group keys as columns
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
   
    
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.values]
    return agg


def make_summary_rows(df_runs, agg):
    """
    Function that builds summary rows to append to CSV file.
    Each summary row corresponds to one group (method and constraint setting) 
    and reports the aggregated metrics (mean and std) computed on successful runs only.

    :arg(df_runs, agg):
        df_runs = pandas DataFrame containing per-run results (one row per run)
        agg = pandas DataFrame produced by aggregate(), containing <metric>_mean and <metric>_std

    :return: pandas DataFrame containing one 'SUMMARY' row per group
    """
    if df_runs is None or len(df_runs) == 0 or agg is None or len(agg) == 0:
        return pd.DataFrame()

    # Recover group column names after the flattening performed in `aggregate()`
    def _col(base):
        # base = group column name
        return base + "_" if (base + "_") in agg.columns else base

    mcol = _col("method")
    ccol = _col("n_constraints_total")
    mlcol = _col("n_ml_req")
    clcol = _col("n_cl_req")

    metric_bases = ["nmi", "cri", "pair_prec", "pair_rec", "pair_f1", "ml_sat", "cl_sat", "time_sec"]

    rows = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S") # timestamp
    
    # Selection of successful runs
    if "failed" in df_runs.columns:
        # select only the rows in the DataFrame where the failed column is 0
        df_success = df_runs[df_runs["failed"] == 0]
    else:
        df_success = df_runs

    for _, arow in agg.iterrows(): # arow = single aggregate row
        method = arow.get(mcol, "")
        total_c = int(arow.get(ccol, 0))
        n_ml_req = int(arow.get(mlcol, 0))
        n_cl_req = int(arow.get(clcol, 0))

        # Count how many successful runs contributed to this aggregate
        mask = (
            (df_success["method"] == method) &
            (df_success["n_constraints_total"] == total_c) &
            (df_success["n_ml_req"] == n_ml_req) &
            (df_success["n_cl_req"] == n_cl_req)
        )
        n_eff = int(mask.sum()) # n_eff counts how many True lines there are
        
        # SUMMARY row
        out = {
            "timestamp": ts,
            "dataset": str(DATASET),
            "dataset_str": str(df_runs["dataset_str"].iloc[0]) if "dataset_str" in df_runs.columns else "",
            "method": method,
            
            "K": int(df_runs["K"].iloc[0]) if "K" in df_runs.columns else np.nan,
            "n": int(df_runs["n"].iloc[0]) if "n" in df_runs.columns else np.nan,
            "run": "SUMMARY",
            "seed_run": "",
            "labeled_frac": float(df_runs["labeled_frac"].iloc[0]) if "labeled_frac" in df_runs.columns else np.nan,
            "n_labeled": "",
            "sampler": str(df_runs["sampler"].iloc[0]) if "sampler" in df_runs.columns else "",
           
            "n_constraints_total": total_c,
            "n_ml_req": n_ml_req,
            "n_cl_req": n_cl_req,
            "n_ml": "",
            "n_cl": "",
            "n_runs_effective": n_eff,
          
            "failed": "",
            "error": "",
        }

        for base in metric_bases:
            out[f"{base}_mean"] = float(arow.get(f"{base}_mean", np.nan))
            out[f"{base}_std"] = float(arow.get(f"{base}_std", np.nan))

        rows.append(out)

    return pd.DataFrame(rows)


# PLOTTING 

def plot_curves(agg, out_png, title, method):
    """
    Function that plots performance curves (with error bars) with respect to the
    total number of pairwise constraints and saves the resulting figure to disk.

    :arg(agg, out_png, title, method):
        agg = aggregated DataFrame containing mean/std columns per metric
        out_png = output path of the PNG figure to be saved
        title  = plot title prefix 
        method = method name string used in the title 
    """
    xcol = "n_constraints_total_" if "n_constraints_total_" in agg.columns else "n_constraints_total"
    x = agg[xcol].to_numpy()

    def yerr(col):
        y = agg[f"{col}_mean"].to_numpy()
        e = agg[f"{col}_std"].to_numpy()
        e = np.nan_to_num(e, nan=0.0)
        return y, e

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    for col, lab in [
        ("nmi", "NMI"),
        ("cri", "CRI (Adjusted Rand)"),
        ("pair_f1", "Pairwise F1"),
        ("ml_sat", "ML satisfaction"),
        ("cl_sat", "CL satisfaction"),
    ]:
        y, e = yerr(col)
        ax.errorbar(x, y, yerr=e, marker="o", linestyle="-", label=lab)

    ax.set_xlabel("#constraints (ML+CL)")
    ax.set_ylabel("Score")
    ax.set_title(f"{title} — {method}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# OUTPUT UTILITIES

def _base_dir():
    """
    Function that returns the base directory where result files are saved.
    
    :return: path to the base directory (string)       
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def save_results_csv(rows, out_path):
    """
    Function that saves per-run rows plus SUMMARY rows (mean ± std) into a 
    single CSV file.

    :arg(rows, out_path):
        rows = list of dictionaries, one per run
        out_path = path of the output CSV file to be written

    :return: aggregated DataFrame computed from successful runs
    """
    if not rows:
        return pd.DataFrame()

    # per-run dataframe 
    df_runs = pd.DataFrame(rows)

    # aggregate successful runs 
    agg_df = aggregate(df_runs[df_runs["failed"] == 0])

    # build summary rows
    summary_df = make_summary_rows(df_runs, agg_df)

    df_out = pd.concat([df_runs, summary_df], ignore_index=True, sort=False)

    # Automatically fix integer-like float columns (avoid .0) 
    '''
    for col in df_out.columns:
        if pd.api.types.is_float_dtype(df_out[col]): # if the column is type float
            s = df_out[col].dropna()
            if not s.empty and np.allclose(s, np.round(s)):
                df_out[col] = df_out[col].astype("Int64")
    '''
    # write CSV
    df_out.to_csv(out_path, sep=";", index=False, na_rep="", float_format="%.3f")

    return agg_df


# MAIN

def main():
    # Load & prepare
    X0, y0 = load_openml_dataset(DATASET)
    
    if isinstance(DATASET, str) and DATASET.lower() in OPENML_IDS:
        ds_id_str = f"{DATASET.lower()}({OPENML_IDS[DATASET.lower()]})"
    else:
        ds_id_str = str(DATASET).strip()
    
    # Preprocessing
    X, y = prepare_X_y(X0, y0, standardize=STANDARDIZE)
   
    vals = np.unique(y)
    print("Unique y:", vals)
    print("y dtype:", y.dtype)
    K = int(len(vals))
    n = int(len(X))
    print("K:", K, "class counts:", np.bincount(y)) # counting by class

    # Params 
    spectral_params = dict(
        graph_type=str(GRAPH_TYPE),
        n_neighbors=int(K_NN),
        gamma=GAMMA,
        normalized=bool(NORMALIZED_LAPLACIAN),
        delta=float(DELTA),
        beta_search=bool(BETA_SEARCH),
        beta_search_iters=int(BETA_SEARCH_ITERS),
        w_ml=float(W_ML),
        w_cl=float(W_CL),
        kmeans_n_init=int(KMEANS_N_INIT),
    )
   
    mpck_params = dict(
        w=float(MPCK_W),
        max_iter=int(MPCK_MAX_ITER),
        eps0=1e-5,
        reg_max_tries=20,
        reg_growth=10.0,
    )

    # Methods
    m = str(METHOD).strip().lower()
    if m == "both":
        methods = ["spectral", "mpck"]
    elif m in ("spectral", "mpck"):
        methods = [m]
    else:
        raise ValueError('METHOD must be "spectral", "mpck", or "both"')
    
    # Setup results folders + basic seed
    base_dir = _base_dir()
    results_dir = os.path.join(base_dir, "Results_Clustering")
    os.makedirs(results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    rng0 = default_rng(SEED)
    rows = []
    
    # Print info
    print(f"Dataset: {ds_id_str} | n={n} | K={K}")
    print(f"Method(s): {methods} | labeled_frac={LABELED_FRAC} | runs={N_RUNS}")
   
    # Experiment Loop
    for total_c in CONSTRAINTS_GRID:
        n_ml_req = int(round(total_c * float(ML_RATIO))) # number of must-links required by ML_RATIO
        n_cl_req = int(total_c - n_ml_req)

        for r in range(N_RUNS):
            # random seed for the current run
            run_seed = int(rng0.integers(0, 2**32 - 1))
            rng = default_rng(run_seed)
            # This seed controls labeled sampling and constraint generation
            
            # fraction of labeled points on which to generate constraints
            n_labeled = max(2, int(round(float(LABELED_FRAC) * n)))
            
            # ensure at least 2 per class (when possible)
            min_needed = 2 * K
            n_labeled = min(n, max(n_labeled, min_needed))
           
            picked = []
            
            # 2 per class
            for c in vals:  # vals contains the unique classes
                # find the indices of all samples that belong to class c
                idx_c = np.flatnonzero(y == c)
                if len(idx_c) >= 2:
                   chosen = rng.choice(idx_c, size=2, replace=False)
                   picked.extend(chosen.tolist())
            
            picked = np.array(sorted(set(picked)), dtype=int)
            
            # fill the remaining labeled quota at random
            if len(picked) < n_labeled:
                remaining = np.setdiff1d(np.arange(n), picked, assume_unique=False)
                extra = rng.choice(remaining, size=(n_labeled - len(picked)), replace=False)
                labeled_idx = np.concatenate([picked, extra])
            else:
                labeled_idx = picked
            
            # constraint sampling
            if total_c == 0:
                must_link, cannot_link = [], []
            else:
                must_link, cannot_link = sample_constraints_class_balanced(
                        y, labeled_idx, n_ml_req, n_cl_req, rng)
           
            print("n_ML:", len(must_link))
            print("n_CL:", len(cannot_link))
            
            # consistency of sampled constraints with true labels
            if len(must_link) > 0:
                ml_purity = np.mean([y[i] == y[j] for i, j in must_link])
            else:
                ml_purity = np.nan
            
            if len(cannot_link) > 0:
                cl_purity = np.mean([y[i] != y[j] for i, j in cannot_link])
            else:
                cl_purity = np.nan
            
            print("ML purity:", ml_purity)
            print("CL purity:", cl_purity)


            # run methods 
            for method in methods:
                t0 = time.time()
                failed = False
                labels = None
                extras = {}

                try:
                    if method == "spectral":
                        labels, extras = run_spectral_ssc(
                            X, K, must_link, cannot_link, spectral_params, run_seed
                        )
                    else:
                        labels, extras = run_mpckmeans(
                            X, K, must_link, cannot_link, mpck_params, run_seed
                        )
                except Exception as e:
                    failed = True
                    extras["error"] = str(e)
                if failed:
                    print("ERROR:", extras.get("error", ""))

                elapsed = time.time() - t0
                
                
                # metrics 
                if not failed:
                    nmi = float(normalized_mutual_info_score(y, labels))
                    cri = float(adjusted_rand_score(y, labels))
                    pair_prec, pair_rec, pair_f1 = pairwise_scores(y, labels)

                    ml_v, cl_v = constraint_violations(labels, must_link, cannot_link)
                    ml_sat = 1.0 - (ml_v / max(1, len(must_link))) if len(must_link) else 1.0
                    cl_sat = 1.0 - (cl_v / max(1, len(cannot_link))) if len(cannot_link) else 1.0
                else:
                    nmi = cri = pair_prec = pair_rec = pair_f1 = np.nan
                    ml_sat = cl_sat = np.nan
                    ml_v = cl_v = None

                # save row 
                rows.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset": str(DATASET),
                    "dataset_str": ds_id_str,
                    "method": method,

                    "K": K,
                    "n": n,
                    "run": r + 1,
                    "seed_run": run_seed,
                    "labeled_frac": float(LABELED_FRAC),
                    "n_labeled": int(n_labeled),

                    "n_constraints_total": int(total_c),
                    "n_ml_req": int(n_ml_req),
                    "n_cl_req": int(n_cl_req),
                    "n_ml": int(len(must_link)),
                    "n_cl": int(len(cannot_link)),

                    "nmi": nmi,
                    "cri": cri,
                    "pair_prec": pair_prec,
                    "pair_rec": pair_rec,
                    "pair_f1": pair_f1,
                    "ml_sat": ml_sat,
                    "cl_sat": cl_sat,
                    "ml_violated": ml_v,
                    "cl_violated": cl_v,

                    "time_sec": float(elapsed),
                    "failed": int(failed),

                    # params snapshot
                    "n_neighbors": int(spectral_params["n_neighbors"]) if method == "spectral" else np.nan,
                    "graph_type": str(spectral_params["graph_type"]) if method == "spectral" else "",
                    "normalized_laplacian": bool(spectral_params.get("normalized")) if method == "spectral" else np.nan,
                    "delta": float(spectral_params.get("delta")) if method == "spectral" else np.nan,
                    "beta_search": bool(spectral_params.get("beta_search")) if method == "spectral" else np.nan,
                    "w_ml": float(spectral_params["w_ml"]) if method == "spectral" else np.nan,
                    "w_cl": float(spectral_params["w_cl"]) if method == "spectral" else np.nan,
                    "kmeans_n_init": int(spectral_params["kmeans_n_init"]) if method == "spectral" else np.nan,

                    "mpck_w": float(mpck_params["w"]) if method == "mpck" else np.nan,
                    "mpck_max_iter": int(mpck_params["max_iter"]) if method == "mpck" else np.nan,

                    "error": extras.get("error", ""),
                })

                print(
                    f"[{method:7s} | C={total_c:4d} | run {r+1:02d}/{N_RUNS}] failed={failed}  "
                    f"NMI={nmi if not failed else 'NA'}  "
                    f"CRI={cri if not failed else 'NA'}  "
                    f"F1={pair_f1 if not failed else 'NA'}  "
                    f"MLsat={ml_sat if not failed else 'NA'}  "
                    f"CLsat={cl_sat if not failed else 'NA'}  "
                    f"t={elapsed:.2f}s"
                )

    # Save results
    out_csv = os.path.join(results_dir, f"results_{DATASET}_{METHOD}_{stamp}.csv")   
    agg_df = save_results_csv(rows, out_csv)

    base_title = f"Dataset {ds_id_str} (K={K})"
    plots = []
    if len(methods) == 1:
        out_png = os.path.join(results_dir, f"clustering_plot_{DATASET}_{METHOD}_{stamp}.png")
        plot_curves(agg_df, out_png, base_title, methods[0])
        plots.append(out_png)
    else:
        for method in methods:
            sub = agg_df[agg_df["method"] == method].copy()
            out_png = os.path.join(results_dir, f"clustering_plot_{DATASET}_{method}_{stamp}.png")
            plot_curves(sub, out_png, base_title, method)
            plots.append(out_png)

    print("\nSaved:")
    print("  csv :", out_csv)
    for p in plots:
        print("  plot:", p)


if __name__ == "__main__":
    main()
