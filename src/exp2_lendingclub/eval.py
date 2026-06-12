import os
import time
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

def load_test(test_csv):
    t0 = time.time()
    df = pd.read_csv(test_csv)
    y = df["loan_status"]
    X = df.drop(columns=["id","loan_status"], errors="ignore")
    logging.info(f"✅ Loaded {test_csv} in {time.time()-t0:.2f}s")
    return X, y, df

def generate_preds(df, group_features, top_models, model_dir, nf=5):
    preds = []
    for grp, feats in group_features.items():
        Xg = df[feats].copy()
        for mname, _ in top_models[grp]:
            fold_probs = []
            for f in range(1, nf+1):
                mdl = joblib.load(os.path.join(model_dir, f"{grp}_{mname}_fold_{f}.pkl"))
                fold_probs.append(mdl.predict_proba(Xg)[:,1])
            preds.append(np.mean(fold_probs, axis=0).reshape(-1,1))
    return np.hstack(preds)

def plot_and_save(y_true, y_prob, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(15,12))
    # ROC
    plt.subplot(2,2,1)
    fpr,tpr,_ = roc_curve(y_true,y_prob)
    plt.plot(fpr,tpr,label=f"AUC={roc_auc_score(y_true,y_prob):.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    # PR
    plt.subplot(2,2,2)
    pr,rc,_ = precision_recall_curve(y_true,y_prob)
    plt.plot(rc,pr,label=f"PR AUC={average_precision_score(y_true,y_prob):.3f}")
    plt.title("Precision–Recall"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    # Calibration
    plt.subplot(2,2,3)
    pt,pp = calibration_curve(y_true,y_prob,n_bins=10)
    plt.plot(pp,pt,'o-'); plt.plot([0,1],[0,1],'k--')
    plt.title("Calibration"); plt.xlabel("Pred"); plt.ylabel("True")
    # Confusion
    plt.subplot(2,2,4)
    y_pred = (y_prob>=0.5).astype(int)
    cm = confusion_matrix(y_true,y_pred)
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.title("Confusion"); plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"full_evaluation.png"))
    plt.close()

def evaluate(test_csv, model_dir, out_dir):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger()

    # 1) load
    X_test, y_test, df_test = load_test(test_csv)

    # feature‐groups (same as train.py)
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

    # 2) load top_models & stacking_meta
    top_models = joblib.load(os.path.join(model_dir,"top_models.pkl"))
    meta       = joblib.load(os.path.join(model_dir,"stacking_meta_model.pkl"))

    # 3) base-model stacking preds
    stacked = generate_preds(df_test, group_features, top_models, model_dir)

    # 4) meta-model final
    probs = meta.predict_proba(stacked)[:,1]

    # 5) metrics + plots
    plot_and_save(y_test, probs, out_dir)
    report = classification_report(y_test,(probs>=0.5).astype(int), output_dict=True)
    logger.info("📋 Classification Report:\n" + pd.DataFrame(report).transpose().to_string())
    os.makedirs(out_dir, exist_ok=True)
    # save per-sample
    pd.DataFrame({
        "true":y_test, "prob":probs, "pred": (probs >= 0.5).astype(int)
    }).to_csv(os.path.join(out_dir,"final_predictions.csv"), index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate LendingClub subset")
    p.add_argument("--test-csv",  default="test_processed.csv")
    p.add_argument("--model-dir", default="enhanced_models")
    p.add_argument("--out-dir",   default="evaluation_plots")
    args = p.parse_args()

    evaluate(args.test_csv, args.model_dir, args.out_dir)