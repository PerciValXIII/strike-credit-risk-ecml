# src/diagnostics/cmi.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer

# ---------------------------
# Core CMI computation
# ---------------------------

def _pca_compress(X: np.ndarray, n_components: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Low-rank compress a group's feature matrix for stable MI estimation.
    Keeps min(n_components, num_features). If features < n_components, no problem.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_comp = min(n_components, X.shape[1])
    if n_comp <= 0:
        return X
    pca = PCA(n_components=n_comp, random_state=random_state)
    return pca.fit_transform(X)

def _pairwise_mi_continuous(A: np.ndarray, B: np.ndarray, random_state: int = 42, k: int = 5) -> float:
    """
    Approximate MI(A;B) for multi-dim A,B by averaging MI over up to k random 1D pairs.
    We use mutual_info_regression(feature(s)=A_i, target=B_j) as a symmetric surrogate.
    """
    rng = np.random.default_rng(random_state)
    a_cols = A.shape[1]
    b_cols = B.shape[1]
    if a_cols == 0 or b_cols == 0:
        return 0.0

    k = max(1, min(k, a_cols, b_cols))
    a_idx = rng.choice(a_cols, size=k, replace=(a_cols < k))
    b_idx = rng.choice(b_cols, size=k, replace=(b_cols < k))

    scores = []
    for i, j in zip(a_idx, b_idx):
        # MI(A_i ; B_j): treat B_j as target, A_i as 1D feature
        try:
            mi1 = mutual_info_regression(A[:, [i]], B[:, j], random_state=random_state)[0]
        except ValueError:
            mi1 = 0.0
        # Symmetrize (optional): also MI(B_j ; A_i)
        try:
            mi2 = mutual_info_regression(B[:, [j]], A[:, i], random_state=random_state)[0]
        except ValueError:
            mi2 = 0.0
        scores.append(0.5 * (mi1 + mi2))
    return float(np.mean(scores)) if scores else 0.0

def cmi_matrix_from_groups(
    X_by_group: Dict[str, np.ndarray],
    y: np.ndarray,
    *,
    pca_components: int = 5,
    random_state: int = 42,
    pairs_k: int = 5,
) -> pd.DataFrame:
    """
    Compute a Groups x Groups matrix of Conditional Mutual Information:
        I(Ga ; Gb | Y) = sum_y P(Y=y) * I(Ga ; Gb | Y=y)
    We estimate I(Ga;Gb | Y=y) on the subset with label y, after PCA compression.

    Args:
        X_by_group: dict of {group_name: 2D ndarray (n_samples x d_g)} for the SAME samples order
        y: 1D binary array (0/1) aligned with rows in group matrices
        pca_components: PCA components per group for MI stability
        pairs_k: number of random 1D pairs to average inside MI surrogate

    Returns:
        DataFrame (groups x groups) with CMI values (nats; relative scale is what matters).
    """
    # Basic checks
    group_names = list(X_by_group.keys())
    n = None
    for g, Xg in X_by_group.items():
        if Xg.ndim != 2:
            raise ValueError(f"Group '{g}' must be 2D; got shape {Xg.shape}.")
        if n is None:
            n = Xg.shape[0]
        elif Xg.shape[0] != n:
            raise ValueError(f"All groups must have same #rows. Group '{g}' has {Xg.shape[0]} vs {n}.")
    if len(y) != n:
        raise ValueError(f"y length {len(y)} must match group matrices rows {n}.")

    # Pre-compress groups with PCA (once)
    Xc = {}
    for g, Xg in X_by_group.items():
        # Replace non-finite with column means
        Xg = Xg.copy()
        if not np.all(np.isfinite(Xg)):
            col_means = np.nanmean(np.where(np.isfinite(Xg), Xg, np.nan), axis=0)
            inds = ~np.isfinite(Xg)
            Xg[inds] = np.take(col_means, np.where(inds)[1])
        Xc[g] = _pca_compress(Xg, n_components=pca_components, random_state=random_state)

    # Class weights
    classes, counts = np.unique(y, return_counts=True)
    weights = counts / counts.sum()

    # Compute CMI
    G = len(group_names)
    CMI = np.zeros((G, G), dtype=float)
    for i, gi in enumerate(group_names):
        for j, gj in enumerate(group_names):
            if i == j:
                CMI[i, j] = 0.0
                continue
            mi_sum = 0.0
            for cls, w in zip(classes, weights):
                mask = (y == cls)
                A = Xc[gi][mask]
                B = Xc[gj][mask]
                mi_cls = _pairwise_mi_continuous(A, B, random_state=random_state, k=pairs_k)
                mi_sum += float(w) * float(mi_cls)
            CMI[i, j] = mi_sum

    return pd.DataFrame(CMI, index=group_names, columns=group_names)
