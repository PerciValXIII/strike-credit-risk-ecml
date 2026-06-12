# STRIKE Experiment: Home Credit Default Risk

This module applies **STRIKE**, a feature-group-aware stacking framework, to the **Home Credit Default Risk** dataset. The Home Credit dataset is a large-scale real-world credit scoring benchmark with hundreds of engineered predictors and a highly imbalanced binary default target.

This folder contains the full Home Credit experiment pipeline, including:

- feature engineering
- model training and stacking
- ablation studies
- diagnostics used to analyze group structure

---

## Objectives

The Home Credit experiment is used to:

- evaluate STRIKE on a high-dimensional real-world default prediction problem
- study whether semantically defined feature groups improve stacking performance
- reproduce the main Home Credit results reported in the paper
- support ablation and diagnostic analysis for the grouping assumptions used by STRIKE

---

## Folder Structure

### `feature_engineering/`
Contains scripts used to create engineered feature tables from the raw Home Credit data.

### `model_training/`
Contains the main STRIKE training pipeline, including base-model training, out-of-fold prediction generation, and meta-model fitting.

### `ablation_studies/`
Contains scripts for the Home Credit ablation experiments, including:
- no-grouping baseline
- grouping-based ablations
- auxiliary grouping logic and logging helpers

See `ablation_studies/README.md` for details.

### `run_diagnostics.py`
Runs the Home Credit diagnostics pipeline used to study cross-group dependence structure.

Currently, this script computes:

- **Conditional Mutual Information (CMI)** between feature groups conditioned on the target

The outputs are saved under:

```bash
outputs/diagnostics/homecredit/
````

### `logs/`

Stores experiment logs for the main STRIKE Home Credit pipeline.

### `data/`

Stores raw and intermediate Home Credit data files required by the experiment.

---

## Data Preparation

Before running the pipeline, download the required raw CSV files from the shared data source and place them inside:

```bash
src/exp3_homecredit/data/
```

At minimum, the Home Credit workflow expects the raw application and source tables required for feature engineering. Intermediate engineered CSVs will also be written into this same folder.

Important files used in the main pipeline include:

* `application_train.csv`
* `deq_features_level1.csv`
* `vintage_features_1.csv`

---

## Running the Home Credit Experiment

### Step 1: Feature engineering

Generate the engineered feature tables:

```bash
python src/exp3_homecredit/feature_engineering/sm_deq_feature_creation.py
python src/exp3_homecredit/feature_engineering/sm_vin_feature_creation.py
```

This produces the engineered group-specific feature files used by STRIKE.

---

### Step 2: Run the main STRIKE pipeline

Train the base models and stacking meta-model:

```bash
python src/exp3_homecredit/model_training/model_stacking_run.py
```

This pipeline:

* preprocesses the engineered datasets
* trains base learners within each feature group
* generates out-of-fold predictions
* trains the final stacking meta-model
* logs execution details to the Home Credit logs folder

---

### Step 3: Run diagnostics

To compute the Home Credit grouping diagnostics:

```bash
python src/exp3_homecredit/run_diagnostics.py
```

This currently generates:

* `cmi_matrix.csv`
* `cmi_heatmap.png`
* `diagnostics_summary.json`

under:

```bash
outputs/diagnostics/homecredit/
```

These diagnostics are intended to help analyze whether the manually defined feature groups are approximately conditionally independent given the target.

---

### Step 4: Run ablation studies

For ablation experiments, use the scripts under:

```bash
src/exp3_homecredit/ablation_studies/
```

Please refer to:

```bash
src/exp3_homecredit/ablation_studies/README.md
```

for exact instructions and environment-variable usage.

---

## Notes

* Ensure all required CSV files are placed in `src/exp3_homecredit/data/`
* Logs for the main Home Credit STRIKE run are saved under `src/exp3_homecredit/logs/`
* Diagnostic outputs are saved under `outputs/diagnostics/homecredit/`
* Ablation-specific logs are saved inside `src/exp3_homecredit/ablation_studies/logs/`

---

