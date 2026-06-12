# src/exp3_homecredit/ablation_studies/run_ablation_no_grouping.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from exp3_homecredit.ablation_studies.utils_logging import (
    get_ablation_logger,
    log_stage
)

from exp3_homecredit.model_training.base_model_utils import train_base_models_with_oof


# =====================================================================
#  HELPER: minimal preprocessing (NO OHE)
# =====================================================================
def preprocess_numeric_only(df, keep_target=False):
    df = df.drop_duplicates()

    if "SK_ID_PREV" in df.columns:
        df = df.drop(columns=["SK_ID_PREV"])

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df = df[numeric_cols]

    # Remove TARGET unless explicitly kept
    if "TARGET" in df.columns and not keep_target:
        df = df.drop(columns=["TARGET"])

    df = df.replace([np.inf, -np.inf], np.nan).fillna(-999)
    df = df.astype(float)
    return df


def scale_train_test(train_df, test_df):
    train = train_df.copy()
    test  = test_df.copy()

    special = ["SK_ID_CURR", "TARGET"]
    features = [c for c in train.columns if c not in special]

    scaler = MinMaxScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features]  = scaler.transform(test[features])

    return train, test


# =====================================================================
#  MAIN SCRIPT
# =====================================================================
if __name__ == "__main__":

    log_dir = Path(__file__).resolve().parent / "logs"
    logger = get_ablation_logger("AblationA_no_grouping", log_dir)

    logger.info("🚀 Starting Ablation A (NO GROUPING, NUMERIC-ONLY, NO OHE)\n")

    data_dir = Path(__file__).resolve().parents[1] / "data"

    # -------------------------------------------------------------
    # 1. LOAD RAW DATA
    # -------------------------------------------------------------
    with log_stage(logger, "Loading RAW datasets"):
        demog_raw = pd.read_csv(data_dir / "application_train.csv")
        deq_raw   = pd.read_csv(data_dir / "deq_features_level1.csv")
        vin_raw   = pd.read_csv(data_dir / "vintage_features_1.csv")

        logger.info(f"Demog RAW: {demog_raw.shape}")
        logger.info(f"Deq RAW:   {deq_raw.shape}")
        logger.info(f"Vin RAW:   {vin_raw.shape}")

    # -------------------------------------------------------------
    # 2. MINIMAL PREPROCESS (NO OHE)
    # -------------------------------------------------------------
    with log_stage(logger, "Minimal preprocessing (NO OHE)"):
        demog_clean = preprocess_numeric_only(demog_raw, keep_target=True)
        deq_clean   = preprocess_numeric_only(deq_raw, keep_target=False)
        vin_clean   = preprocess_numeric_only(vin_raw, keep_target=False)

        logger.info(f"Demog cleaned: {demog_clean.shape}")
        logger.info(f"Deq cleaned:   {deq_clean.shape}")
        logger.info(f"Vin cleaned:   {vin_clean.shape}")

    # -------------------------------------------------------------
    # 3. MERGE INTO FULL FEATURE MATRIX
    # -------------------------------------------------------------
    with log_stage(logger, "Merging datasets"):
        merged = (
            demog_clean
            .merge(deq_clean, on="SK_ID_CURR", how="inner")
            .merge(vin_clean, on="SK_ID_CURR", how="inner")
        ).reset_index(drop=True)

        logger.info(f"Merged shape: {merged.shape}")

    # -------------------------------------------------------------
    # 4. TRAIN/TEST SPLIT
    # -------------------------------------------------------------
    with log_stage(logger, "Train/Test Split"):
        train_raw, test_raw = train_test_split(
            merged,
            test_size=0.25,
            stratify=merged["TARGET"],
            random_state=42
        )
        logger.info(f"Train RAW: {train_raw.shape}")
        logger.info(f"Test RAW:  {test_raw.shape}")

    # -------------------------------------------------------------
    # 5. SCALE TRAIN → TRANSFORM TEST
    # -------------------------------------------------------------
    with log_stage(logger, "Scaling (fit on train → transform test)"):
        train_prep, test_prep = scale_train_test(train_raw, test_raw)

        logger.info(f"Scaled Train: {train_prep.shape}")
        logger.info(f"Scaled Test:  {test_prep.shape}")

    # -------------------------------------------------------------
    # 6. RUN MODEL ZOO USING OOF CV (FOR MODEL SELECTION)
    # -------------------------------------------------------------
    with log_stage(logger, "Training Base Models on ALL features (OOF CV)"):
        oof_df, _ = train_base_models_with_oof(
            train_prep,
            feature_group="all",
            model_save_dir="outputs/ablation_no_grouping/",
            logger=logger
        )

    # Identify best OOF model
    best_model_name = oof_df.columns[1]  # first column is SK_ID_CURR
    algo = best_model_name.split("_")[-1]
    logger.info(f"🏆 Best model according to OOF: {algo.upper()}")

    # -------------------------------------------------------------
    # 7. RETRAIN BEST MODEL ON FULL TRAIN
    # -------------------------------------------------------------
    with log_stage(logger, f"Retraining best model ({algo}) on full train"):

        X_train = train_prep.drop(columns=["SK_ID_CURR", "TARGET"])
        y_train = train_prep["TARGET"]

        if algo == "xgb":
            best_model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        elif algo == "lgb":
            best_model = lgb.LGBMClassifier(random_state=42)
        elif algo == "logreg":
            best_model = LogisticRegression(max_iter=1000, random_state=42)
        elif algo == "rf":
            best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif algo == "knn":
            best_model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"Unknown algo: {algo}")

        best_model.fit(X_train, y_train)

    # -------------------------------------------------------------
    # 8. FINAL TEST EVALUATION
    # -------------------------------------------------------------
    with log_stage(logger, "Final Test Evaluation"):

        X_test = test_prep.drop(columns=["SK_ID_CURR", "TARGET"])
        y_test = test_prep["TARGET"]

        preds = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, preds)

        logger.info(f"🎯 Ablation A — FINAL TEST AUC: {test_auc:.4f}")
        print(f"\n🎯 Ablation A — FINAL TEST AUC: {test_auc:.4f}\n")

    logger.info("🎉 Ablation A completed successfully.")
