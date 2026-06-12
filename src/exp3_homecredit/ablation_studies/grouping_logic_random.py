#grouping_logic_random.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


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


def _check_no_feature_name_collisions(
    demog_df: pd.DataFrame,
    deq_df: pd.DataFrame,
    vin_df: pd.DataFrame
) -> None:
    demog_feats = set(_get_feature_cols(demog_df))
    deq_feats = set(_get_feature_cols(deq_df))
    vin_feats = set(_get_feature_cols(vin_df))

    overlap = (demog_feats & deq_feats) | (demog_feats & vin_feats) | (deq_feats & vin_feats)
    if overlap:
        sample = sorted(list(overlap))[:25]
        raise ValueError(
            "Feature name collision across raw groups detected. "
            f"Example overlapping columns: {sample}"
        )


def load_raw_group_data(
    demog_path: str,
    deq_path: str,
    vin_path: str
) -> RawGroupData:
    demog = pd.read_csv(demog_path)
    deq = pd.read_csv(deq_path)
    vin = pd.read_csv(vin_path)

    _assert_required_cols(demog, "DEMOG raw")
    _assert_required_cols(deq, "DEQ raw")
    _assert_required_cols(vin, "VIN raw")

    _check_no_feature_name_collisions(demog, deq, vin)

    return RawGroupData(demog=demog, deq=deq, vin=vin)


def build_master_table(raw: RawGroupData) -> pd.DataFrame:
    """
    Merge all three raw tables on SK_ID_CURR.
    TARGET is taken from demog, but consistency is verified.
    """
    demog = raw.demog.copy()
    deq = raw.deq.copy()
    vin = raw.vin.copy()

    # TARGET consistency checks
    tmp = demog[[ID_COL, TARGET_COL]].merge(
        deq[[ID_COL, TARGET_COL]],
        on=ID_COL,
        how="inner",
        suffixes=("_d", "_q")
    )
    if not (tmp[f"{TARGET_COL}_d"].values == tmp[f"{TARGET_COL}_q"].values).all():
        raise ValueError("TARGET mismatch between demog and deq for some SK_ID_CURR.")

    tmp = demog[[ID_COL, TARGET_COL]].merge(
        vin[[ID_COL, TARGET_COL]],
        on=ID_COL,
        how="inner",
        suffixes=("_d", "_v")
    )
    if not (tmp[f"{TARGET_COL}_d"].values == tmp[f"{TARGET_COL}_v"].values).all():
        raise ValueError("TARGET mismatch between demog and vin for some SK_ID_CURR.")

    master = demog[[ID_COL, TARGET_COL] + _get_feature_cols(demog)].merge(
        deq[[ID_COL] + _get_feature_cols(deq)],
        on=ID_COL,
        how="outer"
    ).merge(
        vin[[ID_COL] + _get_feature_cols(vin)],
        on=ID_COL,
        how="outer"
    )

    return master


def get_grouping_map_random_mixed(
    raw: RawGroupData,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create 3 NEW random groups that deliberately mix features from
    demog, deq, and vin.

    Logic:
    - shuffle features within each original source
    - split each source into 3 chunks
    - randomly assign those 3 chunks to the 3 new groups

    This guarantees that every new group gets features from every source
    (up to +/-1 feature due to array splitting).
    """
    rng = np.random.default_rng(seed)

    original_source_features = {
        "demog": _get_feature_cols(raw.demog),
        "deq": _get_feature_cols(raw.deq),
        "vin": _get_feature_cols(raw.vin),
    }

    # New synthetic random groups.
    # Names kept as demog/deq/vin only because downstream STRIKE expects 3 files.
    new_groups = {
        "demog": [],
        "deq": [],
        "vin": [],
    }
    new_group_names = ["demog", "deq", "vin"]

    for source_name, feat_list in original_source_features.items():
        feat_list = list(rng.permutation(feat_list))

        # Split this source into 3 nearly equal chunks
        chunks = [list(chunk) for chunk in np.array_split(feat_list, 3)]

        # Randomly assign the 3 chunks to the 3 new groups
        assigned_targets = list(rng.permutation(new_group_names))

        for chunk, target_group in zip(chunks, assigned_targets):
            new_groups[target_group].extend(chunk)

    # Final shuffle within each new group for cleanliness
    for g in new_groups:
        new_groups[g] = list(rng.permutation(new_groups[g]))

    # Safety checks
    all_original_features = (
        original_source_features["demog"]
        + original_source_features["deq"]
        + original_source_features["vin"]
    )
    all_new_features = new_groups["demog"] + new_groups["deq"] + new_groups["vin"]

    if len(all_new_features) != len(all_original_features):
        raise ValueError("Feature count mismatch after random grouping.")

    if len(set(all_new_features)) != len(all_original_features):
        raise ValueError("Duplicate or missing features detected in random grouping.")

    return new_groups


def summarize_grouping_sources(
    grouping_map: Dict[str, List[str]],
    raw: RawGroupData
) -> Dict[str, Dict[str, int]]:
    """
    Optional helper for logging:
    shows how many original demog/deq/vin features ended up inside each new group.
    """
    source_sets = {
        "demog": set(_get_feature_cols(raw.demog)),
        "deq": set(_get_feature_cols(raw.deq)),
        "vin": set(_get_feature_cols(raw.vin)),
    }

    summary: Dict[str, Dict[str, int]] = {}
    for new_group, feats in grouping_map.items():
        feat_set = set(feats)
        summary[new_group] = {
            "from_demog": len(feat_set & source_sets["demog"]),
            "from_deq": len(feat_set & source_sets["deq"]),
            "from_vin": len(feat_set & source_sets["vin"]),
            "total": len(feats),
        }
    return summary


def materialize_grouped_raw_tables(
    raw: RawGroupData,
    strategy: str = "random",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Returns:
        (demog_df_new, deq_df_new, vin_df_new, grouping_map)

    Even though the returned file names are demog/deq/vin for STRIKE compatibility,
    the feature contents are newly created random mixed groups.
    """
    strategy = strategy.lower().strip()
    if strategy != "random":
        raise ValueError(
            f"This file only supports strategy='random'. Got strategy='{strategy}'."
        )

    master = build_master_table(raw)
    grouping_map = get_grouping_map_random_mixed(raw=raw, seed=seed)

    demog_new = master[[ID_COL, TARGET_COL] + grouping_map["demog"]].copy()
    deq_new = master[[ID_COL, TARGET_COL] + grouping_map["deq"]].copy()
    vin_new = master[[ID_COL, TARGET_COL] + grouping_map["vin"]].copy()

    return demog_new, deq_new, vin_new, grouping_map