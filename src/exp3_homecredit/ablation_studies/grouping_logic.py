# src/exp3_homecredit/ablation_studies/grouping_logic.py

# This file:

# reads the 3 raw datasets
# builds a unified feature table aligned on SK_ID_CURR
# creates a grouping_map for strategies: manual | random | corr | mi
# returns three new DataFrames shaped exactly like STRIKE expects:
# each contains SK_ID_CURR, TARGET, and that group’s assigned features




from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scipy.cluster.hierarchy import linkage, fcluster


ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"


@dataclass
class RawGroupData:
    demog: pd.DataFrame
    deq: pd.DataFrame
    vin: pd.DataFrame


def _assert_required_cols(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in [ID_COL, TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in [ID_COL, TARGET_COL]]


def _check_no_feature_name_collisions(demog_df: pd.DataFrame, deq_df: pd.DataFrame, vin_df: pd.DataFrame) -> None:
    demog_feats = set(_get_feature_cols(demog_df))
    deq_feats   = set(_get_feature_cols(deq_df))
    vin_feats   = set(_get_feature_cols(vin_df))

    overlap = (demog_feats & deq_feats) | (demog_feats & vin_feats) | (deq_feats & vin_feats)
    if overlap:
        # Important: If we auto-rename, your MANUAL baseline changes.
        # So we fail fast and ask you to fix collisions upstream.
        sample = sorted(list(overlap))[:25]
        raise ValueError(
            "Feature name collision across raw groups detected (must be resolved to keep ablation fair). "
            f"Example overlapping columns: {sample}"
        )


def load_raw_group_data(
    demog_path: str,
    deq_path: str,
    vin_path: str
) -> RawGroupData:
    demog = pd.read_csv(demog_path)
    deq   = pd.read_csv(deq_path)
    vin   = pd.read_csv(vin_path)

    _assert_required_cols(demog, "DEMOG raw")
    _assert_required_cols(deq, "DEQ raw")
    _assert_required_cols(vin, "VIN raw")

    _check_no_feature_name_collisions(demog, deq, vin)

    return RawGroupData(demog=demog, deq=deq, vin=vin)


def build_master_table(raw: RawGroupData) -> Tuple[pd.DataFrame, List[str]]:
    """
    Outer-merge features across groups on SK_ID_CURR.
    TARGET is taken from demog by default; we also verify consistency where available.
    """
    demog = raw.demog.copy()
    deq = raw.deq.copy()
    vin = raw.vin.copy()

    # Validate TARGET consistency for overlapping IDs (if any mismatch -> error)
    # (outer merge allows missing TARGET in some tables; but you said each has TARGET)
    tmp = demog[[ID_COL, TARGET_COL]].merge(deq[[ID_COL, TARGET_COL]], on=ID_COL, how="inner", suffixes=("_d", "_q"))
    if not (tmp[f"{TARGET_COL}_d"].values == tmp[f"{TARGET_COL}_q"].values).all():
        raise ValueError("TARGET mismatch between demog and deq for some SK_ID_CURR.")
    tmp = demog[[ID_COL, TARGET_COL]].merge(vin[[ID_COL, TARGET_COL]], on=ID_COL, how="inner", suffixes=("_d", "_v"))
    if not (tmp[f"{TARGET_COL}_d"].values == tmp[f"{TARGET_COL}_v"].values).all():
        raise ValueError("TARGET mismatch between demog and vin for some SK_ID_CURR.")

    master = demog[[ID_COL, TARGET_COL] + _get_feature_cols(demog)].merge(
        deq[[ID_COL] + _get_feature_cols(deq)], on=ID_COL, how="outer"
    ).merge(
        vin[[ID_COL] + _get_feature_cols(vin)], on=ID_COL, how="outer"
    )

    feature_cols = [c for c in master.columns if c not in [ID_COL, TARGET_COL]]
    return master, feature_cols


def _template_group_sizes(raw: RawGroupData) -> Dict[str, int]:
    return {
        "demog": len(_get_feature_cols(raw.demog)),
        "deq":   len(_get_feature_cols(raw.deq)),
        "vin":   len(_get_feature_cols(raw.vin)),
    }


def _encode_for_grouping(master: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Create a numeric matrix for corr/MI computation without doing STRIKE preprocessing.
    - numeric: fillna(-999)
    - categorical: factorize to ints, fillna(-1)
    """
    X_parts = []
    for c in feature_cols:
        s = master[c]
        if pd.api.types.is_numeric_dtype(s):
            X_parts.append(s.replace([np.inf, -np.inf], np.nan).fillna(-999).to_numpy().reshape(-1, 1))
        else:
            # factorize categoricals deterministically
            codes, _ = pd.factorize(s.astype("object"), sort=True)
            codes = np.where(pd.isna(s), -1, codes)
            X_parts.append(codes.reshape(-1, 1))
    X = np.hstack(X_parts)
    return X


def get_grouping_map_manual(raw: RawGroupData) -> Dict[str, List[str]]:
    return {
        "demog": _get_feature_cols(raw.demog),
        "deq":   _get_feature_cols(raw.deq),
        "vin":   _get_feature_cols(raw.vin),
    }


def get_grouping_map_random(
    all_features: List[str],
    template_sizes: Dict[str, int],
    seed: int = 42
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(all_features)

    grouping = {}
    idx = 0
    for g in ["demog", "deq", "vin"]:
        size = template_sizes[g]
        grouping[g] = list(shuffled[idx:idx + size])
        idx += size
    return grouping


def get_grouping_map_corr(
    X_enc: np.ndarray,
    feature_cols: List[str],
    n_groups: int = 3
) -> Dict[str, List[str]]:
    # correlation between columns
    corr = np.corrcoef(X_enc, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)

    dist = 1.0 - corr
    Z = linkage(dist, method="average")
    labels = fcluster(Z, t=n_groups, criterion="maxclust")  # 1..n_groups

    groups = {k: [] for k in ["demog", "deq", "vin"]}
    keys = list(groups.keys())

    for feat, lab in zip(feature_cols, labels):
        groups[keys[(lab - 1) % n_groups]].append(feat)
    return groups


def get_grouping_map_mi(
    X_enc: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
    template_sizes: Dict[str, int],
    seed: int = 42
) -> Dict[str, List[str]]:
    mi = mutual_info_classif(X_enc, y, random_state=seed)
    ranked = [f for _, f in sorted(zip(mi, feature_cols), reverse=True)]

    grouping = {}
    idx = 0
    for g in ["demog", "deq", "vin"]:
        size = template_sizes[g]
        grouping[g] = ranked[idx:idx + size]
        idx += size
    return grouping


def materialize_grouped_raw_tables(
    raw: RawGroupData,
    strategy: str,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Returns (demog_df_new, deq_df_new, vin_df_new, grouping_map)
    Each df has columns: SK_ID_CURR, TARGET, <assigned_features...>
    """
    strategy = strategy.lower().strip()
    template_sizes = _template_group_sizes(raw)

    # manual: return originals untouched to preserve exact baseline
    if strategy == "manual":
        grouping_map = get_grouping_map_manual(raw)
        return raw.demog.copy(), raw.deq.copy(), raw.vin.copy(), grouping_map

    master, feature_cols = build_master_table(raw)
    all_features = feature_cols

    X_enc = _encode_for_grouping(master, feature_cols)
    y = master[TARGET_COL].fillna(0).astype(int).to_numpy()

    if strategy == "random":
        grouping_map = get_grouping_map_random(all_features, template_sizes, seed=seed)
    elif strategy == "corr":
        grouping_map = get_grouping_map_corr(X_enc, feature_cols, n_groups=3)
        # enforce exact group sizes by trimming/padding deterministically (reviewer fairness)
        grouping_map = _force_group_sizes(grouping_map, all_features, template_sizes, seed=seed)
    elif strategy == "mi":
        grouping_map = get_grouping_map_mi(X_enc, y, feature_cols, template_sizes, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use one of: manual, random, corr, mi")

    demog_new = master[[ID_COL, TARGET_COL] + grouping_map["demog"]].copy()
    deq_new   = master[[ID_COL, TARGET_COL] + grouping_map["deq"]].copy()
    vin_new   = master[[ID_COL, TARGET_COL] + grouping_map["vin"]].copy()

    return demog_new, deq_new, vin_new, grouping_map


def _force_group_sizes(
    grouping_map: Dict[str, List[str]],
    all_features: List[str],
    template_sizes: Dict[str, int],
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Corr clustering may yield uneven group sizes. To keep ablation fair,
    we enforce exact sizes by:
    - trimming oversized groups
    - filling undersized groups from leftover pool
    """
    rng = np.random.default_rng(seed)

    # Flatten current assignment
    assigned = set()
    for g in grouping_map:
        assigned.update(grouping_map[g])

    leftover = [f for f in all_features if f not in assigned]
    leftover = list(rng.permutation(leftover))

    fixed = {g: list(grouping_map[g]) for g in ["demog", "deq", "vin"]}

    # Trim
    for g in ["demog", "deq", "vin"]:
        size = template_sizes[g]
        if len(fixed[g]) > size:
            fixed[g] = fixed[g][:size]

    # Recompute assigned after trimming
    assigned2 = set()
    for g in fixed:
        assigned2.update(fixed[g])

    # Add back any trimmed-but-unassigned features into leftover pool
    # (ensure we still have all_features coverage)
    leftover2 = [f for f in all_features if f not in assigned2]
    leftover2 = list(rng.permutation(leftover2))

    # Fill undersized
    ptr = 0
    for g in ["demog", "deq", "vin"]:
        size = template_sizes[g]
        if len(fixed[g]) < size:
            need = size - len(fixed[g])
            fixed[g].extend(leftover2[ptr:ptr + need])
            ptr += need

    return fixed
