# src/model_training/orthodox_stacking_run.py

import pandas as pd
import logging
import os
import time
from contextlib import contextmanager
from src.exp3_homecredit.model_training.base_model_utils import train_base_models_with_oof
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib

# ─── Logging Setup ─────────────────────────────────────────────────────────────
os.makedirs("outputs/logs", exist_ok=True)
log_file = "outputs/logs/orthodox_model_training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── Context Manager for Timing ─────────────────────────────────────────────────
@contextmanager
def log_stage(name: str):
    logger.info(f"🔄 Starting: {name}")
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"✅ Completed: {name} in {elapsed:.2f}s\n")

# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("🚀 Starting orthodox stacking pipeline")

    # 1) Load each feature‐group dataset
    with log_stage("Loading Processed Datasets"):
        demog_df = pd.read_csv("data/processed/demog_features_baseline_ready.csv")
        deq_df   = pd.read_csv("data/processed/deq_features_baseline_ready.csv")
        vin_df   = pd.read_csv("data/processed/vintage_features_baseline_ready.csv")

    # 2) Merge them *horizontally* on SK_ID_CURR
    with log_stage("Concatenating All Features"):
        # drop duplicate TARGET columns after the first
        deq_df  = deq_df.drop(columns=["TARGET"])
        vin_df  = vin_df.drop(columns=["TARGET"])
        data_all = (
            demog_df
            .merge(deq_df, on="SK_ID_CURR", how="inner")
            .merge(vin_df, on="SK_ID_CURR", how="inner")
        )

    # 3) Train all 5 base models on the full feature matrix
    with log_stage("Orthodox Base Models Training"):
        # 'all' labels the group name in the saved model filenames
        all_preds, all_target = train_base_models_with_oof(
            data_all, 
            feature_group="all", 
            logger=logger
        )

    # 4) Train a single meta‐learner on those OOF preds
    with log_stage("Orthodox Meta Model Training"):
        X_meta = all_preds.drop(columns=["SK_ID_CURR"])
        y_meta = all_target["TARGET"]

        meta = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(meta, X_meta, y_meta, cv=5, scoring="roc_auc")
        logger.info(f"[ORTHODOX META] Logistic Regression AUC (CV): {scores.mean():.4f}")

        meta.fit(X_meta, y_meta)
        joblib.dump(meta, "outputs/models/orthodox_meta_logistic_model.pkl")

    logger.info("🎯 Orthodox stacking pipeline completed.")
