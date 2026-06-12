#!/usr/bin/env python
import os
import sys
import time
import argparse
import logging

# allow imports from your package
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../src')))

# core data & model I/O
import joblib
import numpy as np
import pandas as pd

# optional plotting
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn metrics
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, balanced_accuracy_score, f1_score,
    log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve

def run_evaluation(model_dir: str, output_dir: str):
    """
    1) Load test CSV
    2) Load saved base + meta‐models from model_dir
    3) Compute metrics (AUC, PR‐AUC, confusion matrix, etc.)
    4) Save plots/CSVs under output_dir
    """


    # 0. Configure logging globally
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()

    # Configuration
    MODEL_DIR = model_dir
    NFOLDS    = 5
    PLOT_DIR  = output_dir
    os.makedirs(PLOT_DIR, exist_ok=True)


    def load_test_data():
        logger.info("🔄 Loading processed test data from disk")
        t0 = time.time()
        path = os.path.join(model_dir, "test_data_processed.joblib")
        data = joblib.load(path)
        X_test_full              = data["X_test_full"]
        y_test_full              = data["y_test_full"]
        processed_group_dfs_test = data["processed_group_dfs_test"]
        logger.info(f"✅ Loaded test data in {time.time()-t0:.2f}s")
        return X_test_full, y_test_full, processed_group_dfs_test

    def generate_top_predictions(processed_group_dfs_test, top_models):
        logger.info("🔄 Starting: Generating base‑model predictions on test set")
        t0 = time.time()
        preds_list = []
        for grp, df_grp in processed_group_dfs_test.items():
            Xg = df_grp.drop(columns=["class"], errors="ignore")
            for model_name, _ in top_models[grp]:
                fold_preds = []
                for f in range(1, NFOLDS+1):
                    path = os.path.join(MODEL_DIR, f"{grp}_{model_name}_fold_{f}.pkl")
                    model = joblib.load(path)
                    fold_preds.append(model.predict_proba(Xg)[:,1])
                preds_list.append(np.mean(fold_preds, axis=0).reshape(-1,1))
        stacked = np.hstack(preds_list)
        logger.info(f"✅ Completed: Generating base‑model predictions in {time.time() - t0:.2f}s")
        return stacked

    def generate_evaluation_plots(y_true, y_prob, metrics):
        plt.figure(figsize=(15, 12))
        # ROC
        plt.subplot(2,2,1)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.3f}')
        plt.plot([0,1],[0,1],'k--')
        plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
        # Precision-Recall
        plt.subplot(2,2,2)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, label=f'PR AUC = {metrics["pr_auc"]:.3f}')
        plt.title("Precision–Recall"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
        # Calibration
        plt.subplot(2,2,3)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, 'o-')
        plt.plot([0,1],[0,1],'k--')
        plt.title("Calibration Curve"); plt.xlabel("Predicted"); plt.ylabel("True")
        # Confusion
        plt.subplot(2,2,4)
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "full_evaluation.png"))
        plt.show()
        plt.close()

    def comprehensive_evaluation(y_true, y_prob):
        logger.info("🔄 Starting: Computing evaluation metrics & plots")
        t0 = time.time()
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {
            'confusion_matrix':    confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'roc_auc':             roc_auc_score(y_true, y_prob),
            'pr_auc':              average_precision_score(y_true, y_prob),
        }
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        metrics['optimal_threshold'] = thr[np.argmax(tpr - fpr)]
        generate_evaluation_plots(y_true, y_prob, metrics)
        logger.info(f"✅ Completed: Evaluation in {time.time() - t0:.2f}s")
        return metrics

    def main():
        logger.info("🚀 Starting evaluation pipeline")
        t_pipeline = time.time()


        X_test_full, y_test_full, processed_group_dfs_test = load_test_data()


        tm_path = os.path.join(MODEL_DIR, "top_models.joblib")
        logger.info(f"🔄 Loading top_models from {tm_path}")
        top_models = joblib.load(tm_path)
        # 3) Base-model stacking
        stacked = generate_top_predictions(processed_group_dfs_test, top_models)

        # 4) Meta-model final predictions
        logger.info("🔄 Starting: Meta‑model final predictions")
        t0 = time.time()
        meta = joblib.load(os.path.join(MODEL_DIR, "stacking_meta_model.pkl"))
        final_probs = meta.predict_proba(stacked)[:,1]
        logger.info(f"✅ Completed: Meta‑model final predictions in {time.time() - t0:.2f}s")

        # 5) Compute metrics & plots
        metrics = comprehensive_evaluation(y_test_full, final_probs)

        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        logger.info("📋 Classification Report:\n" + report_df.to_string())
        # 6) Save & log final results
        results_df = pd.DataFrame({
            'true_label':      y_test_full,
            'predicted_prob':  final_probs,
            'predicted_label': (final_probs >= metrics['optimal_threshold']).astype(int)
        })
        results_df.to_csv(os.path.join(PLOT_DIR, "final_predictions.csv"), index=False)

        logger.info("🎯 Final Evaluation Metrics:")
        logger.info(f"    ROC AUC:           {metrics['roc_auc']:.4f}")
        logger.info(f"    PR AUC:            {metrics['pr_auc']:.4f}")
        logger.info(f"    Optimal threshold: {metrics['optimal_threshold']:.4f}")
        logger.info(f"🎉 Evaluation pipeline completed successfully in {time.time() - t_pipeline:.2f}s")


        
        NFOLDS    = 5

        # 1) Prepare a container for each classifier
        base_models = ["XGBoost","GBDT","AdaBoost","RandomForest","LightGBM"]
        preds = {m: np.zeros(len(y_test_full)) for m in base_models}

        # 2) For each model, for each group, load all folds, average their proba, and slot into preds[m]
        for m in base_models:
            for grp, df_grp in processed_group_dfs_test.items():
                # indices of this group in the full test set
                idx = df_grp.index

                # collect per‑fold probabilities
                fold_probs = []
                for f in range(1, NFOLDS+1):
                    path = os.path.join(MODEL_DIR, f"{grp}_{m}_fold_{f}.pkl")
                    mdl  = joblib.load(path)
                    Xg   = df_grp.drop(columns=["class"], errors="ignore")
                    fold_probs.append(mdl.predict_proba(Xg)[:,1])
                # average across folds
                avg_prob = np.mean(fold_probs, axis=0)

                # place into the correct slots in the full test‑set vector
                preds[m][idx] = avg_prob

        # 3) Compute all six metrics for each model
        results = []
        for m in base_models:
            p = preds[m]
            y_pred = (p >= 0.5).astype(int)
            res = {
                "Method":     "Our Method",
                "Base":       m if m!="RandomForest" else "RF",
                "ACC":        accuracy_score(y_test_full, y_pred),
                "BA":         balanced_accuracy_score(y_test_full, y_pred),
                "AUC":        roc_auc_score(y_test_full, p),
                "Log loss":   log_loss(y_test_full, np.vstack([1-p, p]).T),
                "Brier":      brier_score_loss(y_test_full, p)
            }
            results.append(res)


        df = pd.DataFrame(results)
        print(df)
    main()


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Evaluate STRIKE on Polish bankruptcy data"
    )
    p.add_argument(
        '--model-dir', required=True,
        help='Folder where train.py dumped the models'
    )
    p.add_argument(
    '--output-dir', default='evaluation_plots',
    help='Where to save evaluation outputs')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)


    run_evaluation(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )