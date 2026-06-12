# src/exp3_homecredit/ablation_studies/run_ablation_grouping_logic.py

from __future__ import annotations

import os
import shutil
import sys
import time
import subprocess
from pathlib import Path

import pandas as pd

from src.exp3_homecredit.ablation_studies.utils_logging import get_ablation_logger

## Use this when all grouping logic to be tested
# from src.exp3_homecredit.ablation_studies.grouping_logic import (
#     load_raw_group_data,
#     materialize_grouped_raw_tables,
# )

# Use this to test only random grouping logic
from src.exp3_homecredit.ablation_studies.grouping_logic_random import summarize_grouping_sources

from src.exp3_homecredit.ablation_studies.grouping_logic_random import (
    load_raw_group_data,
    materialize_grouped_raw_tables,
)

# ---------- Paths ----------
THIS_FILE = Path(__file__).resolve()
EXP_ROOT = THIS_FILE.parents[1]          # .../src/exp3_homecredit
DATA_DIR = EXP_ROOT / "data"
ABL_DIR = THIS_FILE.parent
ABL_LOG_DIR = ABL_DIR / "logs"
ABL_LOG_DIR.mkdir(parents=True, exist_ok=True)

STRIKE_SCRIPT = EXP_ROOT / "model_training" / "model_stacking_run.py"
STRIKE_LOG = EXP_ROOT / "logs" / "strike_homecredit_stacking.log"

RAW_FILES = {
    "demog": "application_train.csv",
    "deq":   "deq_features_level1.csv",
    "vin":   "vintage_features_1.csv",
}

# outputs/models are overwritten by STRIKE
OUTPUTS_DIR = EXP_ROOT.parents[2] / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"


# ======================================================
# Utilities
# ======================================================

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ======================================================
# Row Sampling (NEW)
# ======================================================

def stratified_sample_ids(
    demog_df: pd.DataFrame,
    n_rows: int,
    seed: int,
) -> set[int]:
    """
    Stratified sampling by TARGET using demog table.
    Returns a set of SK_ID_CURR to keep.
    """
    df = demog_df[["SK_ID_CURR", "TARGET"]].dropna()
    n_rows = min(n_rows, len(df))

    pos = df[df["TARGET"] == 1]
    neg = df[df["TARGET"] == 0]

    pos_frac = len(pos) / len(df)
    pos_n = int(n_rows * pos_frac)
    neg_n = n_rows - pos_n

    pos_ids = pos.sample(n=pos_n, random_state=seed)["SK_ID_CURR"]
    neg_ids = neg.sample(n=neg_n, random_state=seed)["SK_ID_CURR"]

    return set(pd.concat([pos_ids, neg_ids]).tolist())


# ======================================================
# Backup / Restore
# ======================================================

def _backup_inputs(run_id: str, logger) -> Path:
    backup_dir = ABL_DIR / "_tmp_backup" / run_id
    backup_dir.mkdir(parents=True, exist_ok=True)

    for fname in RAW_FILES.values():
        _safe_copy(DATA_DIR / fname, backup_dir / fname)

    if STRIKE_LOG.exists():
        _safe_copy(STRIKE_LOG, backup_dir / STRIKE_LOG.name)

    if MODELS_DIR.exists():
        shutil.copytree(MODELS_DIR, backup_dir / "models_backup", dirs_exist_ok=True)

    logger.info(f"✅ Backed up raw inputs → {backup_dir}")
    return backup_dir


def _restore_inputs(backup_dir: Path, logger) -> None:
    for fname in RAW_FILES.values():
        _safe_copy(backup_dir / fname, DATA_DIR / fname)

    bk_log = backup_dir / STRIKE_LOG.name
    if bk_log.exists():
        _safe_copy(bk_log, STRIKE_LOG)

    bk_models = backup_dir / "models_backup"
    if bk_models.exists():
        if MODELS_DIR.exists():
            shutil.rmtree(MODELS_DIR)
        shutil.copytree(bk_models, MODELS_DIR, dirs_exist_ok=True)

    logger.info("✅ Restored original raw inputs.")


# ======================================================
# STRIKE Execution
# ======================================================

def _write_ablated_raws(demog_df, deq_df, vin_df, logger):
    demog_df.to_csv(DATA_DIR / RAW_FILES["demog"], index=False)
    deq_df.to_csv(DATA_DIR / RAW_FILES["deq"], index=False)
    vin_df.to_csv(DATA_DIR / RAW_FILES["vin"], index=False)
    logger.info("✅ Wrote ablated raw CSVs (STRIKE filenames preserved).")


def _run_strike(logger) -> int:
    logger.info(f"🚀 Running STRIKE unchanged: {STRIKE_SCRIPT}")
    proc = subprocess.run(
        [sys.executable, str(STRIKE_SCRIPT)],
        cwd=str(EXP_ROOT.parents[2]),
    )
    return proc.returncode


# ======================================================
# Main
# ======================================================

def main():
    # ---- Controls ----
    strategy = os.environ.get("GROUPING_STRATEGY", "manual").lower()
    grouping_seed = int(os.environ.get("GROUPING_SEED", "42"))

    # NEW: fixed row sampling (same rows across all runs)
    n_rows = int(os.environ.get("ROW_SAMPLE_SIZE", "50000"))
    row_seed = int(os.environ.get("ROW_SAMPLE_SEED", "123"))

    run_id = f"grouping_{strategy}_seed{grouping_seed}_rows{n_rows}_{_timestamp()}"

    logger = get_ablation_logger(
        name=f"ablation_{run_id}",
        log_dir=ABL_LOG_DIR,
    )

    logger.info(
        f"🚀 Starting ablation_grouping_logic | "
        f"strategy={strategy} | grouping_seed={grouping_seed} | "
        f"rows={n_rows} | row_seed={row_seed}"
    )

    backup_dir = _backup_inputs(run_id, logger)

    try:
        # ---- Load raw data ----
        raw = load_raw_group_data(
            demog_path=str(DATA_DIR / RAW_FILES["demog"]),
            deq_path=str(DATA_DIR / RAW_FILES["deq"]),
            vin_path=str(DATA_DIR / RAW_FILES["vin"]),
        )

        # ---- Row subsampling (NEW) ----
        sample_ids = stratified_sample_ids(
            raw.demog, n_rows=n_rows, seed=row_seed
        )

        raw.demog = raw.demog[raw.demog["SK_ID_CURR"].isin(sample_ids)].copy()
        raw.deq   = raw.deq[raw.deq["SK_ID_CURR"].isin(sample_ids)].copy()
        raw.vin   = raw.vin[raw.vin["SK_ID_CURR"].isin(sample_ids)].copy()

        logger.info(
            f"Using stratified subsample: {len(sample_ids)} rows "
            f"(pos rate preserved)"
        )

        # ---- Grouping ablation ----
        demog_new, deq_new, vin_new, grouping_map = materialize_grouped_raw_tables(
            raw=raw,
            strategy=strategy,
            seed=grouping_seed,
        )
        
        source_mix_summary = summarize_grouping_sources(grouping_map, raw)

        logger.info(
            f"Grouping sizes | demog={len(grouping_map['demog'])} | "
            f"deq={len(grouping_map['deq'])} | vin={len(grouping_map['vin'])}"
        )

        logger.info(f"Source-mix summary: {source_mix_summary}")


        _write_ablated_raws(demog_new, deq_new, vin_new, logger)

        # ---- Run STRIKE ----
        rc = _run_strike(logger)
        if rc != 0:
            raise RuntimeError(f"STRIKE failed with return code {rc}")

        # ---- Archive STRIKE log ----
        if STRIKE_LOG.exists():
            dst = ABL_LOG_DIR / f"strike_log_{run_id}.log"
            _safe_copy(STRIKE_LOG, dst)
            logger.info(f"✅ Copied STRIKE log → {dst}")

        logger.info("🎯 ablation_grouping_logic completed successfully.")

    finally:
        _restore_inputs(backup_dir, logger)


if __name__ == "__main__":
    main()
