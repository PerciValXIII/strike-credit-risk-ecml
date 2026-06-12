#!/usr/bin/env python
import os
import sys
import time
import argparse
import logging

# allow imports from src/
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

# Data handling
import pandas as pd
import numpy as np
import joblib

# Visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# ML frameworks
import xgboost as xgb
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Scikit-learn utilities
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Parallelization helper
from joblib import Parallel, delayed


def run_training(input_path: str,output_dir: str):
    """
    1) Load & EDA
    2) Cleaning & preprocessing
    3) Base‐models + meta‐model training
    4) Save models/logs under output_dir
    """

    os.makedirs(output_dir, exist_ok=True)
    start_eda = time.time()  # Start timer for EDA
    # Load the dataset (we now assume the data is in CSV format)
    data = pd.read_csv(input_path)
    df = pd.DataFrame(data)


    # Print basic dataset information
    print("Dataset shape:", df.shape)
    print("Dataset columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

    # Display detailed info and summary statistics
    print("\nData Info:")
    df.info()
    print("\nSummary Statistics:")
    print(df.describe())

    # Check for missing values in each column
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Identify numeric columns for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # # Plot histograms for the first few numeric columns to inspect distributions
    # for col in numeric_cols[:5]:
    #     plt.figure(figsize=(8, 4))
    #     sns.histplot(df[col].dropna(), kde=True)
    #     plt.title(f'Distribution of {col}')
    #     plt.xlabel(col)
    #     plt.ylabel('Frequency')
    #     plt.show()

    # # Optional: Correlation heatmap for numeric features
    # plt.figure(figsize=(12, 10))
    # corr = df[numeric_cols].corr()
    # sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    # plt.title("Correlation Heatmap of Numeric Features")
    # plt.show()

    end_eda = time.time()  # End timer for EDA
    print("Total EDA Execution Time: {:.2f} seconds".format(end_eda - start_eda))
    start_preprocessing = time.time()  # Start overall preprocessing timer

    # Ensure column names are clean
    df.columns = df.columns.str.strip()
    print("Columns after stripping:", df.columns.tolist())

    # Perform train-test split (make sure these variables are global)
    X = df.drop(columns=['class'])
    y = df['class']
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Training set shape:", X_train_full.shape)
    print("Test set shape:", X_test_full.shape)

    # Define the grouping of features by category.
    group_features = {
        "Profitability": ['Attr1', 'Attr6', 'Attr7', 'Attr11', 'Attr12', 'Attr13', 'Attr14',
                          'Attr16', 'Attr18', 'Attr19', 'Attr22', 'Attr23', 'Attr24', 'Attr26',
                          'Attr31', 'Attr35', 'Attr39', 'Attr42', 'Attr48', 'Attr49', 'Attr56'],
        "Leverage":      ['Attr2', 'Attr8', 'Attr10', 'Attr17', 'Attr25', 'Attr38', 'Attr51',
                          'Attr53', 'Attr54', 'Attr59'],
        "Liquidity":     ['Attr3', 'Attr4', 'Attr5', 'Attr28', 'Attr37', 'Attr40', 'Attr46',
                          'Attr50', 'Attr55', 'Attr57'],
        "Efficiency":    ['Attr9', 'Attr15', 'Attr20', 'Attr27', 'Attr30', 'Attr32', 'Attr33',
                          'Attr34', 'Attr36', 'Attr41', 'Attr43', 'Attr44', 'Attr45', 'Attr47',
                          'Attr52', 'Attr58', 'Attr60', 'Attr61', 'Attr62', 'Attr63', 'Attr64'],
        "GrowthSize":    ['Attr21', 'Attr29']
    }

    # Initialize a dictionary to store fitted scalers for each group.
    scalers = {group: None for group in group_features}

    def process_category_df(df_cat, scaler=None, fit_scaler=False, target_col=None, quantile_thresholds=None):
        """
        Process a DataFrame for a given category.
          - One-hot encodes categorical columns.
          - Imputes missing numeric values with -999.
          - Caps outliers using 1st and 99th percentiles.
            * If quantile_thresholds is None, compute them (training data).
            * Otherwise, use the provided thresholds.
          - Scales numeric features using MinMaxScaler.
        Returns the processed DataFrame, the scaler, and the quantile thresholds.
        """
        features_df = df_cat.copy()

        # Separate target column if provided.
        if target_col and target_col in features_df.columns:
            target_series = features_df[target_col]
            features_df = features_df.drop(columns=[target_col])
        else:
            target_series = None

        # One-hot encode categorical columns.
        categorical_cols = features_df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)

        # Identify numeric columns.
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Impute missing numeric values.
        features_df[numeric_cols] = features_df[numeric_cols].fillna(-999)

        # Outlier capping: compute thresholds on training data or use provided thresholds.
        if quantile_thresholds is None:
            quantile_thresholds = {}
            for col in numeric_cols:
                lower = features_df[col].quantile(0.01)
                upper = features_df[col].quantile(0.99)
                quantile_thresholds[col] = (lower, upper)
                features_df[col] = features_df[col].clip(lower, upper)
        else:
            for col in numeric_cols:
                if col in quantile_thresholds:
                    lower, upper = quantile_thresholds[col]
                    features_df[col] = features_df[col].clip(lower, upper)
                else:
                    # Optionally compute new thresholds (or leave as is)
                    lower = features_df[col].quantile(0.01)
                    upper = features_df[col].quantile(0.99)
                    quantile_thresholds[col] = (lower, upper)
                    features_df[col] = features_df[col].clip(lower, upper)

        # Scale numeric columns.
        if numeric_cols:
            if fit_scaler:
                scaler = MinMaxScaler().fit(features_df[numeric_cols])
            features_df[numeric_cols] = scaler.transform(features_df[numeric_cols])

        # Add target back if it was removed.
        if target_series is not None:
            features_df[target_col] = target_series.values
        return features_df, scaler, quantile_thresholds



    start_train_preprocessing = time.time()  # Start timer for training preprocessing

    # ---------------------------
    # Process and Check Training Data by Category
    # ---------------------------
    processed_group_dfs_train = {}
    scalers = {}
    quantile_thresholds_dict = {}

    print("----- Processing Training Data by Category -----")
    for group, feats in group_features.items():
        subset_features = X_train_full[feats].copy()
        subset_target = y_train_full.reset_index(drop=True)
        subset_df = pd.concat([subset_features.reset_index(drop=True), subset_target], axis=1)
        print(f"Before processing for {group}: {subset_df.shape}")

        processed_df, fitted_scaler, quantile_thresholds = process_category_df(
            subset_df, scaler=None, fit_scaler=True, target_col='class', quantile_thresholds=None
        )
        scalers[group] = fitted_scaler
        quantile_thresholds_dict[group] = quantile_thresholds
        processed_group_dfs_train[group] = processed_df
        print(f"After processing for {group}: {processed_df.shape}")

    end_train_preprocessing = time.time()  # End timer for training preprocessing
    print("Training Data Preprocessing Time: {:.2f} seconds".format(end_train_preprocessing - start_train_preprocessing))
    # ---------------------------
    # Process and Check Test Data by Category
    # ---------------------------

    start_test_preprocessing = time.time()  # Start timer for test preprocessing

    processed_group_dfs_test = {}
    print("\n----- Processing Test Data by Category -----")
    for group, feats in group_features.items():
        subset_features = X_test_full[feats].copy()
        subset_target = y_test_full.reset_index(drop=True)
        subset_df = pd.concat([subset_features.reset_index(drop=True), subset_target], axis=1)
        print(f"Before processing for {group}: {subset_df.shape}")

        processed_df, _ , _ = process_category_df(
            subset_df,
            scaler=scalers[group],
            fit_scaler=False,
            target_col='class',
            quantile_thresholds=quantile_thresholds_dict[group]
        )
        processed_group_dfs_test[group] = processed_df
        print(f"After processing for {group}: {processed_df.shape}")

    end_test_preprocessing = time.time()  # End timer for test preprocessing
    print("Test Data Preprocessing Time: {:.2f} seconds".format(end_test_preprocessing - start_test_preprocessing))
    save_path = os.path.join(output_dir, "test_data_processed.joblib")
    joblib.dump({
        "X_test_full": X_test_full,
        "y_test_full": y_test_full,
        "processed_group_dfs_test": processed_group_dfs_test
    }, save_path)
    logging.info(f"✅ Saved processed test data to {save_path}")
    end_preprocessing = time.time()  # End overall preprocessing timer
    print("Total Preprocessing Execution Time: {:.2f} seconds".format(end_preprocessing - start_preprocessing))

    # 0. Configure logging globally
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger()

    # --- Configuration ---
    NFOLDS    = 5
    SEED      = 42
    N_JOBS    = -1
    MODEL_DIR = output_dir

    # --- Model Configurations (unchanged) ---
    model_configs = {
        "XGBoost": {
            "model": xgb.XGBClassifier,
            "params": {
                "learning_rate": 0.05, "n_estimators": 200, "max_depth": 4,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "objective": "binary:logistic", "n_jobs": N_JOBS,
                "random_state": SEED
            },
            "tune_params": {"max_depth": (3, 10), "learning_rate": (0.01,0.3), "n_estimators": (50,300)}
        },
        "GBDT": {
            "model": GradientBoostingClassifier,
            "params": {
                "learning_rate": 0.05, "n_estimators": 200, "max_depth": 4,
                "subsample": 0.8, "random_state": SEED
            },
            "tune_params": {"max_depth": (3, 10), "learning_rate": (0.01,0.3), "n_estimators": (50,300)}
        },
        "AdaBoost": {
            "model": AdaBoostClassifier,
            "params": {"n_estimators":200, "learning_rate":0.05, "random_state":SEED},
            "tune_params": {"n_estimators": (50,300), "learning_rate": (0.01,1.0)}
        },
        "RandomForest": {
            "model": RandomForestClassifier,
            "params": {
                "n_estimators":200, "max_depth":12, "max_features":0.2,
                "min_samples_leaf":2, "n_jobs":N_JOBS, "random_state":SEED
            },
            "tune_params": {"n_estimators": (50,300), "max_depth": (3,20)}
        },
        "LightGBM": {
            "model": LGBMClassifier,
            "params": {
                "learning_rate":0.05, "n_estimators":200, "max_depth":6,
                "num_leaves":31, "subsample":0.8, "colsample_bytree":0.8,
                "n_jobs":N_JOBS, "random_state":SEED
            },
            "tune_params": {"learning_rate": (0.01,0.3), "n_estimators": (50,300), "num_leaves": (16,64)}
        }
    }

    # --- Start overall pipeline ---
    start_total = time.time()
    logger.info("🚀 Starting model stacking pipeline")

    # --- Hyperparameter Tuning Phase ---
    logger.info("🔄 Starting: Hyperparameter Tuning Phase")
    start_tune = time.time()

    def tune_hyperparameters(X, y, cfg):
        def objective(trial):
            params = {}
            for p, (low, high) in cfg["tune_params"].items():
                if isinstance(low, int):
                    params[p] = trial.suggest_int(p, low, high)
                else:
                    params[p] = trial.suggest_float(p, low, high)
            mdl = cfg["model"](**{**cfg["params"], **params})
            return np.mean(cross_val_score(mdl, X, y, cv=3, scoring="roc_auc"))
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        fn = os.path.join(MODEL_DIR, f"{cfg['model'].__name__}_tuning_study.pkl")
        joblib.dump(study, fn)
        logger.info(f"✅ Tuning study saved as: {fn}")
        return study.best_params

    group_model_best = {}
    for grp, df_grp in processed_group_dfs_train.items():
        Xg = df_grp.drop(columns=["class"])
        yg = df_grp["class"]
        group_model_best[grp] = {}
        for mname, cfg in model_configs.items():
            if "tune_params" in cfg:
                logger.info(f" 🔧 Tuning {mname} for {grp}")
                best = tune_hyperparameters(Xg, yg, cfg)
                group_model_best[grp][mname] = best
                cfg["params"].update(best)

    end_tune = time.time()
    logger.info("✅ Completed: Hyperparameter Tuning Phase in %.2f seconds", end_tune-start_tune)

    # --- OOF Predictions Generation Phase ---
    logger.info("🔄 Starting: Base Models OOF Training")
    start_oof = time.time()

    def train_fold(fold, tr_idx, val_idx, X_df, y_df, cfg, grp, mname):
        Xtr, Xval = X_df.iloc[tr_idx], X_df.iloc[val_idx]
        ytr, yval = y_df.iloc[tr_idx], y_df.iloc[val_idx]
        mdl = cfg["model"](**cfg["params"])
        mdl.fit(Xtr, ytr)
        path = os.path.join(MODEL_DIR, f"{grp}_{mname}_fold_{fold+1}.pkl")
        joblib.dump(mdl, path)
        logger.info(f" [fold {fold+1}] Saved model: {grp}_{mname}")
        return val_idx, mdl.predict_proba(Xval)[:,1]

    def get_oof(mname, cfg, X_df, y_df, grp):
        oof = np.zeros(len(X_df))
        kf  = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
        results = Parallel(n_jobs=N_JOBS)(
            delayed(train_fold)(i, ti, vi, X_df, y_df, cfg, grp, mname)
            for i,(ti,vi) in enumerate(kf.split(X_df))
        )
        for idxs, preds in results:
            oof[idxs] = preds
        auc = roc_auc_score(y_df, oof)
        logger.info(f" [{grp}] {mname} FINAL OOF AUC: {auc:.4f}")
        return oof.reshape(-1,1), auc

    model_perf = {grp:{} for grp in processed_group_dfs_train}
    oof_preds   = []
    for grp, df_grp in processed_group_dfs_train.items():
        Xg = df_grp.drop(columns=["class"])
        yg = df_grp["class"]
        for mname, cfg in model_configs.items():
            preds, auc = get_oof(mname, cfg, Xg, yg, grp)
            oof_preds.append((f"{grp}_{mname}", preds))
            model_perf[grp][mname] = auc

    end_oof = time.time()
    logger.info("✅ Completed: Base Models OOF Training in %.2f seconds", end_oof-start_oof)

    # --- Top-3 Selection ---
    logger.info("🔄 Starting: Top Model Selection")
    top_models = {}
    for grp, scores in model_perf.items():
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_models[grp] = top
        logger.info(f" [{grp}] ✅ Top 3 Models: {[m for m,_ in top]}")
    logger.info("✅ Completed: Top Model Selection")

    # --- Stack for Meta-Model ---
    logger.info("🔄 Starting: Stacking top model predictions")
    filtered = []
    for grp, mods in top_models.items():
        for mname,_ in mods:
            for name,p in oof_preds:
                if name == f"{grp}_{mname}":
                    filtered.append(p)
    stacked_train = np.hstack(filtered)
    logger.info("✅ Stacked train shape: %s", stacked_train.shape)
    joblib.dump(top_models, os.path.join(output_dir, "top_models.joblib"))
    # --- Meta-Model Training ---
    logger.info("🔄 Starting: Meta Model Training")
    start_meta = time.time()

    def tune_meta(X, y):
        def obj(trial):
            C = trial.suggest_float("C", 0.001, 1.0)
            mdl = LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=SEED)
            return np.mean(cross_val_score(mdl, X, y, cv=3, scoring="roc_auc"))
        st = optuna.create_study(direction="maximize")
        st.optimize(obj, n_trials=10)
        return st.best_params

    best_meta = tune_meta(stacked_train, y_train_full)
    meta = LogisticRegression(**best_meta, penalty="l1", solver="liblinear", random_state=SEED)
    meta.fit(stacked_train, y_train_full)

    mfn = os.path.join(MODEL_DIR, "stacking_meta_model.pkl")
    joblib.dump(meta, mfn)
    logger.info("✅ Meta-model saved: %s", mfn)

    end_meta = time.time()
    logger.info("✅ Completed: Meta Model Training in %.2f seconds", end_meta-start_meta)

    # --- End of pipeline ---
    end_total = time.time()
    logger.info("🎯 Model stacking pipeline completed successfully in %.2f seconds", end_total-start_total)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Train STRIKE on Polish bankruptcy data"
    )
    p.add_argument(
        '--input-data', required=True,
        help='Path to polish_bankruptcy.csv'
    )
    p.add_argument(
        '--output-dir',
        default='enhanced_models',
        help='Where to save models & logs'
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)


    run_training(
        input_path=args.input_data,
        output_dir=args.output_dir,
    )
