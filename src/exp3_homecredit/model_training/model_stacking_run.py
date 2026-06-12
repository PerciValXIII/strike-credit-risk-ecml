import sys
from pathlib import Path

# Add root directory to sys.path BEFORE any src import
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import logging
import time
import os
from contextlib import contextmanager

from exp3_homecredit.feature_engineering.preprocess_pipeline import Preprocessor
from exp3_homecredit.model_training.base_model_utils import train_base_models_with_oof
from exp3_homecredit.model_training.meta_model import train_meta_model

# --- Setup Paths ---
root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
log_dir = root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
log_file = log_dir / "strike_homecredit_stacking.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Utility Timer ---
@contextmanager
def log_stage(name):
    logger.info(f"🔄 Starting: {name}")
    start = time.time()
    yield
    logger.info(f"✅ Completed: {name} in {time.time() - start:.2f} seconds")

# --- Preprocess Wrapper ---
def preprocess_and_save(input_name, output_name):
    df = pd.read_csv(data_dir / input_name)
    df_processed = Preprocessor(df).run()
    df_processed.to_csv(data_dir / output_name, index=False)
    return df_processed

# --- Main Pipeline ---
if __name__ == "__main__":
    logger.info("🚀 Starting STRIKE Home Credit Model Stacking Pipeline")

    with log_stage("Preprocessing DEMOG Features"):
        demog_df = preprocess_and_save("application_train.csv", "demog_features_baseline_ready.csv")

    with log_stage("Preprocessing DEQ Features"):
        deq_df = preprocess_and_save("deq_features_level1.csv", "deq_features_baseline_ready.csv")

    with log_stage("Preprocessing Vintage Features"):
        vin_df = preprocess_and_save("vintage_features_1.csv", "vintage_features_baseline_ready.csv")

    with log_stage("Training Base Models on Demographic Features"):
        demog_preds, _ = train_base_models_with_oof(demog_df, "demog", logger=logger)

    with log_stage("Training Base Models on Delinquency Features"):
        deq_preds, _ = train_base_models_with_oof(deq_df, "deq", logger=logger)

    with log_stage("Training Base Models on Vintage Features"):
        vin_preds, _ = train_base_models_with_oof(vin_df, "vin", logger=logger)

    with log_stage("Creating Meta-Model Training Set"):
        target_df = pd.concat([
            demog_df[["SK_ID_CURR", "TARGET"]],
            deq_df[["SK_ID_CURR", "TARGET"]],
            vin_df[["SK_ID_CURR", "TARGET"]]
        ]).drop_duplicates("SK_ID_CURR")

    with log_stage("Training Meta-Model"):
        train_meta_model(demog_preds, deq_preds, vin_preds, target_df, logger=logger)

    logger.info("🎯 STRIKE Home Credit pipeline completed successfully.")
