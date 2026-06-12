
# STRIKE Experiment: LendingClub Dataset

This module applies **STRIKE**, a multi-model stacking framework, to the **LendingClub consumer lending dataset**—a U.S.-based peer-to-peer lending dataset with over 225,000 loans and approximately 150 features per borrower. The dataset has a moderate class imbalance (~21% defaults).

---

## Objectives

- Validate STRIKE in a real-world consumer lending context.
- Test scalability with a larger dataset and moderately imbalanced target variable.
- Reproduce model-specific metrics as discussed in Section 4.5 of the NeurIPS paper.

---

## Running the Experiment

### Step 0: Prepare Raw Data

Before running any code, download the raw CSV file from the following Google Drive link:

```
https://drive.google.com/drive/folders/1ef8jGMQSni6r4pSW2aMg6c453Y2a5tt6?usp=sharing
```

Place the downloaded csv file - `filtered_data_lendingclub.csv` from `exp2_lendingclub_data` directly inside the `exp2_lendingclub/` directory in the source code. 

---

### Step 1: Preprocess & Train the Model

This step processes the raw CSV data and writes trained model artifacts to the `models/` directory.

```bash
cd strike-credit-risk/exp2_lendingclub

python train.py   --raw-csv filtered_data_lendingclub.csv   --model-dir models
```

### Step 2: Evaluate the Model

This step loads the trained models and generates evaluation plots and metrics under the `evaluation_plots/` directory.

```bash
python eval.py   --model-dir models   --out-dir evaluation_plots
```

---

## Notes

- Make sure `filtered_data_lendingclub.csv` is available in the working directory before training.
- The `models/` directory will contain all trained models, while `evaluation_plots/` will include ROC curves, metrics, and summaries.
- This experiment highlights STRIKE's ability to scale and generalize in a real-world credit risk application.

---

For more details, refer to Section 4.5 of the NeurIPS paper.
