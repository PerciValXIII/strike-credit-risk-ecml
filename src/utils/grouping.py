# src/utils/grouping.py
from __future__ import annotations
from typing import Dict, List
from pathlib import Path
import pandas as pd
import numpy as np

# Import your Preprocessor
from src.exp3_homecredit.feature_engineering.preprocess_pipeline import Preprocessor

# ---- Paths (project-root relative; no hard-coded absolute paths) ----
PROJ_ROOT = Path(__file__).resolve().parents[2]
HC_DATA   = PROJ_ROOT / "src" / "exp3_homecredit" / "data"

DEMOG_CSV = HC_DATA / "application_train.csv"
DEQ_CSV   = HC_DATA / "deq_features_level1.csv"
VIN_CSV   = HC_DATA / "vintage_features_1.csv"

# Columns that must never be returned as features
NON_FEATURE_COLS = {"SK_ID_CURR", "TARGET", "SK_ID_PREV"}

def _load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    """
    Load a CSV and run the project's Preprocessor to ensure we get the exact
    post-processing feature set (OHE, scaling, etc.). Returns the processed DataFrame.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected file not found: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df_proc = Preprocessor(df_raw).run()

    # Drop non-feature columns if they survived (errors='ignore' for robustness)
    to_drop = [c for c in NON_FEATURE_COLS if c in df_proc.columns]
    if to_drop:
        df_proc = df_proc.drop(columns=to_drop, errors="ignore")

    # Sanity: ensure all numeric after Preprocessor
    non_numeric = df_proc.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns remain after preprocessing in {csv_path.name}: {non_numeric}"
        )

    # Sanity: ensure no NaN/inf before we only use column names
    if not np.all(np.isfinite(df_proc.to_numpy())):
        raise ValueError(
            f"Non-finite values remain after preprocessing in {csv_path.name}."
        )

    return df_proc

def _assert_disjoint(groups: Dict[str, List[str]]) -> None:
    """Ensure no column appears in more than one group."""
    seen: Dict[str, str] = {}
    overlaps = []
    for g, cols in groups.items():
        for c in cols:
            if c in seen and seen[c] != g:
                overlaps.append((c, seen[c], g))
            else:
                seen[c] = g
    if overlaps:
        msg = "\n".join([f" - '{c}' appears in groups '{g1}' and '{g2}'" for c, g1, g2 in overlaps])
        raise ValueError(
            "Group columns must be disjoint, but overlaps were found:\n" + msg +
            "\n\nTip: ensure upstream feature files don’t generate identically named columns across groups. "
            "If they must, create a consistent renaming step upstream so names are unique per group."
        )

def _assert_nonempty(groups: Dict[str, List[str]]) -> None:
    empties = [g for g, cols in groups.items() if len(cols) == 0]
    if empties:
        raise ValueError(f"These groups are empty after preprocessing: {empties}. "
                         "Check source CSVs and Preprocessor behavior.")

def get_homecredit_groups() -> Dict[str, List[str]]:
    """
    Returns {group_name: [post-processed feature names]}.
    This intentionally re-runs the project's Preprocessor on each source CSV
    to guarantee column names match the final model inputs used in STRIKE.
    """
    # Process each group’s source file with the same Preprocessor used in training
    df_demog = _load_and_preprocess(DEMOG_CSV)
    df_deq   = _load_and_preprocess(DEQ_CSV)
    df_vin   = _load_and_preprocess(VIN_CSV)

    # Extract post-processed column lists (pure feature names)
    demog_cols = list(df_demog.columns)
    deq_cols   = list(df_deq.columns)
    vin_cols   = list(df_vin.columns)

    # Optional: sort for determinism
    demog_cols = sorted(demog_cols)
    deq_cols   = sorted(deq_cols)
    vin_cols   = sorted(vin_cols)

    groups: Dict[str, List[str]] = {
        "demographics": demog_cols,
        "delinquency":  deq_cols,
        "vintage":      vin_cols,
    }

    # Validate groups
    _assert_nonempty(groups)
    _assert_disjoint(groups)

    return groups

# -------- Convenience: run as a module to preview groups --------
if __name__ == "__main__":
    # If you run:  python -m src.utils.grouping   (from repo root)
    groups = get_homecredit_groups()
    total = sum(len(v) for v in groups.values())
    print(">>> HomeCredit Group Summary (post-Preprocessor)")
    for k, v in groups.items():
        print(f" - {k:12s}: {len(v):4d} features")
    print(f"Total features across groups: {total}")
