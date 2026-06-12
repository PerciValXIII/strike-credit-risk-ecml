
# STRIKE Experiment: Polish Bankruptcy Dataset

This module evaluates **STRIKE**, a multi-model stacking framework, on the **Polish companies bankruptcy dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data). The task involves predicting company-level bankruptcy using 65 financial features from a highly imbalanced dataset of 7,027 observations (only 3.86% labeled as bankrupt).

---

## Objectives

- Demonstrate STRIKE’s robustness in small-sample, high-class-imbalance settings.
- Compare STRIKE against baseline models and orthodox stacking approaches.
- Replicate the results for the Polish dataset as reported in Section 4.1 and 4.5 of the NeurIPS paper.

---

## Running the Experiment

### Step 0: Prepare Raw Data

Before running any code, download the raw CSV file from the following Google Drive link:

```
https://drive.google.com/drive/folders/1ef8jGMQSni6r4pSW2aMg6c453Y2a5tt6?usp=sharing
```

Place the downloaded csv file - `polish_bankruptcy.csv` from `exp1_polish_data` directly inside the `exp1_polish/` directory in the source code. 

---

### Step 1: Train the Model

This step reads the input data and writes all model artifacts to the `models/` directory.

```bash
cd strike-credit-risk/exp1_polish

python train.py   --input-data polish_bankruptcy.csv   --output-dir models
```

### Step 2: Evaluate the Model

This step loads the trained models and generates evaluation plots and metrics under the `evaluation_plots/` directory.

```bash
python eval.py   --model-dir models   --output-dir evaluation_plots
```

---

## Notes

- Ensure that `polish_bankruptcy.csv` is present in the working directory before running `train.py`.
- All plots, evaluation metrics, and CSV summaries will be saved in `evaluation_plots/` after evaluation.
- This experiment is designed to benchmark STRIKE's performance under real-world credit risk modeling conditions.

---

For more details, refer to Section 4.1 and 4.5 of the NeurIPS paper.
