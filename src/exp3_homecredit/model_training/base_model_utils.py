import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

def clean_feature_names(df):
    df.columns = [re.sub(r'[^\w\d_]', '_', col) for col in df.columns]
    return df


def train_base_models_with_oof(data, feature_group, n_splits=5, model_save_dir="outputs/models/", logger=None):

    os.makedirs(model_save_dir, exist_ok=True)

    data = data.copy()
    data = clean_feature_names(data)
    X = data.drop(columns=["SK_ID_CURR", "TARGET"])
    y = data["TARGET"]
    id_col = data["SK_ID_CURR"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "xgb": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "lgb": lgb.LGBMClassifier(random_state=42),
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5)
    }

    auc_scores = {}
    oof_storage = {}
    fold_models = {}

    for name, model in models.items():
        oof_preds = np.zeros(len(data))
        fold_models[name] = []

        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"→ Training {feature_group.upper()} | {name.upper()} | Fold {i+1}/{n_splits}")
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
            val_preds = model_clone.predict_proba(X.iloc[val_idx])[:, 1]

            fold_auc = roc_auc_score(y.iloc[val_idx], val_preds)
            if logger:
                logger.info(f"   AUC for fold {i+1}: {fold_auc:.4f}")

            oof_preds[val_idx] = val_preds

            # Save model for this fold
            model_path = os.path.join(model_save_dir, f"{feature_group}_{name}_fold{i}.pkl")
            joblib.dump(model_clone, model_path)
            fold_models[name].append(model_clone)

        auc = roc_auc_score(y, oof_preds)
        auc_scores[name] = auc
        oof_storage[name] = oof_preds
        if logger:
            logger.info(f"[{feature_group}] {name} FINAL OOF AUC: {auc:.4f}")

    # Select top 3 models by AUC
    top_models = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    if logger:
        logger.info(f"[{feature_group}] ✅ Top 3 Models by AUC: {[m[0] for m in top_models]}\n")


    oof_df = pd.DataFrame({"SK_ID_CURR": id_col})
    for name, _ in top_models:
        oof_df[f"{feature_group}_{name}"] = oof_storage[name]

    return oof_df, data[["SK_ID_CURR", "TARGET"]]
