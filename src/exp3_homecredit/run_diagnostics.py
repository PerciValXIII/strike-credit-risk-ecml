from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.exp3_homecredit.feature_engineering.preprocess_pipeline import Preprocessor
from src.utils.grouping import get_homecredit_groups
from src.diagnostics.cmi import cmi_matrix_from_groups

# -------------------------
# Paths / constants
# -------------------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
HC_DATA   = PROJ_ROOT / "src" / "exp3_homecredit" / "data"

DEMOG_CSV = HC_DATA / "application_train.csv"
DEQ_CSV   = HC_DATA / "deq_features_level1.csv"
VIN_CSV   = HC_DATA / "vintage_features_1.csv"

OUTDIR    = PROJ_ROOT / "outputs" / "diagnostics" / "homecredit"
OUTDIR.mkdir(parents=True, exist_ok=True)

ID_COL     = "SK_ID_CURR"
TARGET_COL = "TARGET"


def _load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df_proc = Preprocessor(df_raw).run()

    non_numeric = df_proc.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns remain after preprocessing in {csv_path.name}: {non_numeric}"
        )

    if not np.all(np.isfinite(df_proc.to_numpy())):
        raise ValueError(
            f"Non-finite values remain after preprocessing in {csv_path.name}."
        )

    return df_proc


def _save_heatmap(df: pd.DataFrame, title: str, path_png: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6.5, 5.5), dpi=300)

    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap=sns.color_palette("mako", as_cmap=True),
        cbar=True,
        linewidths=0.6,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": 10, "color": "white"},
        xticklabels=df.columns,
        yticklabels=df.index,
    )

    ax.set_title(title, fontsize=13, pad=16, fontweight="semibold")
    ax.set_xticklabels(df.columns, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(df.index, fontsize=10, rotation=0)

    plt.tight_layout(pad=1.2)
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()


def _merge_feature_tables(
    demog: pd.DataFrame,
    deq: pd.DataFrame,
    vin: pd.DataFrame,
    *,
    id_col: str,
) -> pd.DataFrame:
    merged = demog.merge(deq, on=id_col, how="inner", suffixes=("", "_DEQ"))
    merged = merged.merge(vin, on=id_col, how="inner", suffixes=("", "_VIN"))
    return merged


def _build_group_matrices(
    merged: pd.DataFrame,
    groups: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    X_by_group = {}
    for gname, cols in groups.items():
        missing = [c for c in cols if c not in merged.columns]
        if missing:
            raise ValueError(
                f"Group '{gname}' columns missing after merge: {missing[:5]} ..."
            )
        X_by_group[gname] = merged[cols].to_numpy()
    return X_by_group


def _drop_label_clones(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    cols_to_drop = [
        c for c in df.columns
        if c == target_col
        or c.startswith(target_col + "_")
        or "target" in c.lower()
    ]
    return df.drop(columns=cols_to_drop, errors="ignore")


def main():
    groups = get_homecredit_groups()

    demog = _load_and_preprocess(DEMOG_CSV)
    deq   = _load_and_preprocess(DEQ_CSV)
    vin   = _load_and_preprocess(VIN_CSV)

    if ID_COL not in demog.columns:
        raise ValueError(f"{ID_COL} not found in processed demographics table.")
    if TARGET_COL not in demog.columns:
        raise ValueError(f"{TARGET_COL} not found in processed demographics table.")
    if ID_COL not in deq.columns or ID_COL not in vin.columns:
        raise ValueError(f"{ID_COL} must exist in deq and vin tables to merge.")

    merged = _merge_feature_tables(demog, deq, vin, id_col=ID_COL)

    y = merged[TARGET_COL].astype(int).to_numpy()

    merged_noleak = _drop_label_clones(merged, TARGET_COL)

    # kept for leakage guard / future diagnostics consistency
    _ = merged_noleak

    X_by_group = _build_group_matrices(merged, groups)

    cmi_df = cmi_matrix_from_groups(
        X_by_group=X_by_group,
        y=y,
        pca_components=5,
        random_state=42,
        pairs_k=5,
    )

    cmi_csv = OUTDIR / "cmi_matrix.csv"
    cmi_png = OUTDIR / "cmi_heatmap.png"
    summary_json = OUTDIR / "diagnostics_summary.json"

    cmi_df.to_csv(cmi_csv, index=True)
    _save_heatmap(cmi_df, "Conditional Mutual Information (Groups | Y)", cmi_png)

    def _offdiag_stats(M: pd.DataFrame) -> dict:
        vals = []
        for i in range(len(M)):
            for j in range(len(M)):
                if i < j:
                    vals.append(M.iloc[i, j])
        return {
            "mean_offdiag": float(np.mean(vals)),
            "median_offdiag": float(np.median(vals)),
            "max_offdiag": float(np.max(vals)),
        }

    out_summary = {
        "cmi_stats": _offdiag_stats(cmi_df),
        "groups": {k: len(v) for k, v in groups.items()},
    }

    with open(summary_json, "w") as f:
        json.dump(out_summary, f, indent=2)

    print("====================================")
    print("Diagnostics complete.")
    print("CMI off-diagonal stats:", out_summary["cmi_stats"])
    print(f"Saved outputs in {OUTDIR}")
    print("====================================")


if __name__ == "__main__":
    main()