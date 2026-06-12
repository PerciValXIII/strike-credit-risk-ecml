# train.py
#!/usr/bin/env python
import os
import time
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed, dump
from sklearn.metrics import roc_auc_score

def preprocess(raw_csv: str, train_csv: str, test_csv: str):
    # ---- Cell 1: Category-Specific Data Preprocessing ----
    raw = pd.read_csv(raw_csv)
    raw.columns = raw.columns.str.strip()

    meta = raw[["id", "loan_status"]].copy()
    df   = raw.drop(columns=["id", "emp_title", "loan_status"], errors="ignore")

    X_train, X_test, meta_train, meta_test = train_test_split(
        df, meta, test_size=0.2, random_state=42, stratify=meta["loan_status"]
    )

    thresh  = int(0.1 * len(X_train))
    high_na = X_train.columns[X_train.isna().sum() > (len(X_train) - thresh)].tolist()
    X_train = X_train.drop(columns=high_na)
    X_test  = X_test.drop(columns=high_na, errors="ignore")

    X_train = X_train.fillna(-999)
    X_test  = X_test.fillna(-999)

    cat_cols   = X_train.select_dtypes(exclude=[np.number]).columns
    low_covors = [
        c for c in cat_cols
        if X_train[c].astype(str).value_counts(normalize=True).nlargest(5).sum() * 100 < 90
    ]
    X_train = X_train.drop(columns=low_covors)
    X_test  = X_test.drop(columns=low_covors, errors="ignore")

    to_dummy = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    X_train  = pd.get_dummies(X_train, columns=to_dummy, drop_first=False)
    X_test   = pd.get_dummies(X_test,  columns=to_dummy, drop_first=False)
    X_test   = X_test.reindex(columns=X_train.columns, fill_value=0)

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler   = MinMaxScaler().fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    train_proc = pd.concat([meta_train.reset_index(drop=True),
                            X_train.reset_index(drop=True)], axis=1)
    test_proc  = pd.concat([meta_test.reset_index(drop=True),
                            X_test.reset_index(drop=True)],  axis=1)

    mapping = {"Fully Paid": 1, "Charged Off": 0}
    train_proc["loan_status"] = train_proc["loan_status"].map(mapping)
    test_proc["loan_status"]  = test_proc["loan_status"].map(mapping)

    train_proc.to_csv(train_csv, index=False)
    test_proc.to_csv(test_csv,   index=False)
    print(f"Saved {train_csv} → {train_proc.shape}")
    print(f"Saved {test_csv}  → {test_proc.shape}")

def train_models(train_csv: str, model_dir: str):
    # ---- Cell 2: Base Model Training, Selection & Meta-Model ----
    os.makedirs(model_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger()

    NFOLDS, SEED, N_JOBS = 5, 42, -1
    train_proc = pd.read_csv(train_csv)
    y_train_full = train_proc["loan_status"]

    # feature‐groups dict (exactly as in your cell)
    group_features = {
    # 1. Core loan terms & borrower capacity
      "Loan_Terms": [
          "loan_amnt", "funded_amnt", "funded_amnt_inv",
          "int_rate", "installment", "dti", "annual_inc", "policy_code"
      ],

      # 2. Credit‐ and account‐level profile
      "Credit_Profile": [
          "delinq_2yrs", "fico_range_low", "fico_range_high",
          "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
          "open_acc", "pub_rec", "total_acc", "acc_now_delinq",
          "collections_12_mths_ex_med", "mths_since_last_major_derog",
          "pub_rec_bankruptcies", "tax_liens"
      ],

      # 3. Utilization & activity metrics
      "Utilization_and_Activity": [
          "revol_bal", "revol_util", "tot_cur_bal", "tot_hi_cred_lim",
          "total_bal_il", "il_util", "total_bal_ex_mort", "total_bc_limit",
          "total_il_high_credit_limit", "bc_open_to_buy", "bc_util",
          "total_rev_hi_lim", "avg_cur_bal", "num_rev_accts",
          "num_rev_tl_bal_gt_0", "pct_tl_nvr_dlq", "percent_bc_gt_75",
          "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
          "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
          "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd",
          "num_tl_90g_dpd_24m", "num_tl_op_past_12m"
      ],

      # 5. One‐hot categorical flags
      "Categorical_Flags": [
          "term_ 36 months", "term_ 60 months",
          "grade_A", "grade_B", "grade_C", "grade_D", "grade_E", "grade_F", "grade_G",
          "home_ownership_ANY", "home_ownership_MORTGAGE", "home_ownership_NONE",
          "home_ownership_OWN", "home_ownership_RENT",
          "verification_status_Not Verified", "verification_status_Source Verified",
          "verification_status_Verified",
          "pymnt_plan_n", "application_type_Individual", "application_type_Joint App",
          "hardship_flag_N", "disbursement_method_Cash", "disbursement_method_DirectPay",
          "debt_settlement_flag_N", "debt_settlement_flag_Y",
          # purpose dummies
          "purpose_car", "purpose_credit_card", "purpose_debt_consolidation",
          "purpose_home_improvement", "purpose_house", "purpose_major_purchase",
          "purpose_medical", "purpose_moving", "purpose_other",
          "purpose_renewable_energy", "purpose_small_business", "purpose_vacation",
          "purpose_wedding",
          # title dummies
          "title_Business", "title_Car financing", "title_Credit card refinancing",
          "title_Debt consolidation", "title_Green loan", "title_Home buying",
          "title_Home improvement", "title_Major purchase", "title_Medical expenses",
          "title_Moving and relocation", "title_Other", "title_Vacation",
          # listing & status
          "initial_list_status_f", "initial_list_status_w"
      ]
  }

    # build per‐group DataFrames
    processed_group_dfs_train = {
        grp: train_proc[ feats + ["loan_status"] ].copy()
        for grp, feats in group_features.items()
    }

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

    def tune_hyperparameters(X, y, cfg):
        def objective(trial):
            params = {}
            for p, (low, high) in cfg["tune_params"].items():
                params[p] = (trial.suggest_int if isinstance(low, int) else trial.suggest_float)(p, low, high)
            mdl = cfg["model"](**{**cfg["params"], **params})
            return np.mean(cross_val_score(mdl, X, y, cv=3, scoring="roc_auc"))
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        fn = os.path.join(model_dir, f"{cfg['model'].__name__}_tuning_study.pkl")
        dump(study, fn)
        logger.info(f"✅ Tuning study saved as: {fn}")
        return study.best_params

    # hyperparameter tuning
    for grp, df_grp in processed_group_dfs_train.items():
        Xg = df_grp.drop(columns=["loan_status"])
        yg = df_grp["loan_status"]
        for mname, cfg in model_configs.items():
            best = tune_hyperparameters(Xg, yg, cfg)
            cfg["params"].update(best)

    # OOF training
    def train_fold(fold, tr_idx, val_idx, X_df, y_df, cfg, grp, mname):
        Xtr, Xval = X_df.iloc[tr_idx], X_df.iloc[val_idx]
        mdl = cfg["model"](**cfg["params"])
        mdl.fit(Xtr, y_df.iloc[tr_idx])
        fn = os.path.join(model_dir, f"{grp}_{mname}_fold_{fold+1}.pkl")
        dump(mdl, fn)
        logger.info(f" [fold {fold+1}] Saved model: {grp}_{mname}")
        return val_idx, mdl.predict_proba(Xval)[:,1]

    oof_preds, model_perf = [], {}
    for grp, df_grp in processed_group_dfs_train.items():
        Xg = df_grp.drop(columns=["loan_status"])
        yg = df_grp["loan_status"]
        model_perf[grp] = {}
        for mname, cfg in model_configs.items():
            oof, auc = np.zeros(len(Xg)), None
            kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
            results = Parallel(n_jobs=N_JOBS)(
                delayed(train_fold)(i, tr, vl, Xg, yg, cfg, grp, mname)
                for i,(tr,vl) in enumerate(kf.split(Xg))
            )
            for idxs, preds in results:
                oof[idxs] = preds
            auc = roc_auc_score(yg, oof)
            model_perf[grp][mname] = auc
            oof_preds.append((f"{grp}_{mname}", oof.reshape(-1,1)))

    # select top 3
    top_models = {
        grp: sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for grp, scores in model_perf.items()
    }
    dump(top_models, os.path.join(model_dir, "top_models.pkl"))
    logger.info("✅ Saved top_models.pkl")

    # stacking
    stacked_train = np.hstack([
        preds for grp, mlist in top_models.items()
        for name, _ in mlist
        for nm, preds in oof_preds if nm == f"{grp}_{name}"
    ])

    # meta-model
    best_meta = tune_hyperparameters(stacked_train, y_train_full, {
        "model": LogisticRegression,
        "params": {"penalty":"l1","solver":"liblinear","random_state":SEED},
        "tune_params": {"C":(0.001,1.0)}
    })
    meta = LogisticRegression(**best_meta, penalty="l1", solver="liblinear", random_state=SEED)
    meta.fit(stacked_train, y_train_full)
    dump(meta, os.path.join(model_dir, "stacking_meta_model.pkl"))
    logger.info("✅ Saved stacking_meta_model.pkl")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess + Train on LendingClub subset")
    p.add_argument("--raw-csv",     default="filtered_data_lendingclub.csv")
    p.add_argument("--train-csv",   default="train_processed.csv")
    p.add_argument("--test-csv",    default="test_processed.csv")
    p.add_argument("--model-dir",   default="enhanced_models")
    args = p.parse_args()

    preprocess(args.raw_csv, args.train_csv, args.test_csv)
    train_models(args.train_csv, args.model_dir)
