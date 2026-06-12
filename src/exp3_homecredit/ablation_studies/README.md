
# Home Credit Ablation Studies

This folder contains the code used to reproduce the **ablation studies** for the Home Credit experiments in STRIKE.

The ablation experiments are designed to answer two main questions:

1. Does STRIKE benefit from **feature grouping** compared with a standard no-grouping setup?
2. Does STRIKE benefit specifically from **meaningful semantic grouping**, rather than from arbitrary/random group partitions?

---

# Folder Overview

## Main scripts

### `run_ablation_grouping_logic.py`
This is the main runner for the **grouping-based ablation**.

It works by:

1. Loading the three raw Home Credit input tables
2. Applying a specified grouping strategy to construct new grouped tables
3. Writing those ablated grouped tables back to the standard STRIKE input filenames
4. Running the unchanged STRIKE pipeline via:

```bash
src/exp3_homecredit/model_training/model_stacking_run.py
````

5. Restoring the original input files after execution

This script is therefore a **wrapper around the main STRIKE pipeline**, not a separate training pipeline.

At present, this script imports from:

* `grouping_logic_random.py`

and is configured for the **random mixed grouping ablation**.

---

### `run_ablation_no_grouping.py`

This script runs the **no-grouping baseline**.

Unlike the grouping ablation runner, this script does **not** call the main STRIKE stacking pipeline.
Instead, it:

1. Loads the three raw data tables
2. Performs minimal numeric-only preprocessing
3. Merges all features into a single feature table
4. Splits data into train/test
5. Trains a small model zoo using OOF CV on the full merged feature set
6. Selects the best-performing base learner
7. Retrains that best learner on the full training set
8. Reports final test AUC

This provides the conventional **all-features-together baseline** used to compare against STRIKE.

---

## Utility / grouping files

### `grouping_logic_random.py`

This file contains the grouping implementation used by `run_ablation_grouping_logic.py`.

It constructs **synthetic random mixed groups** by deliberately redistributing features from the original semantic groups:

* demographic
* delinquency
* vintage

so that each newly formed group contains a mixture of features from all original sources.

This is intended to test whether STRIKE’s performance depends on **semantically coherent group structure**, rather than on arbitrary grouped partitions.

This file also provides:

* raw data loading
* schema validation
* master table creation
* grouping materialization
* source-mix summary logging

---

### `grouping_logic.py`

This file contains a broader grouping framework with support for multiple grouping strategies, including:

* manual
* random
* corr
* mi

However, in the current code state, `run_ablation_grouping_logic.py` is **not importing this file**.
It is retained for transparency and experimentation history, but the active grouping-ablation runner currently uses `grouping_logic_random.py`.

---

### `run_random_grouping_trials.py`

This is an auxiliary script for running repeated random-grouping experiments across multiple seeds, if needed.

It is not required for the primary ablation reproduction unless multiple-seed random grouping trials are specifically being evaluated.

---

### `utils_logging.py`

Logging helper utilities for ablation experiments.

---

### `logs/`

Stores log files produced by the ablation scripts.

---

# Recommended Reproduction Order

## 1. Run the no-grouping baseline

```bash
python src/exp3_homecredit/ablation_studies/run_ablation_no_grouping.py
```

This produces the standard all-features-together baseline.

---

## 2. Run the grouping ablation

Example:

```bash
GROUPING_STRATEGY=random GROUPING_SEED=42 ROW_SAMPLE_SIZE=50000 ROW_SAMPLE_SEED=123 \
python src/exp3_homecredit/ablation_studies/run_ablation_grouping_logic.py
```

This runs the grouping-based ablation while preserving the main STRIKE pipeline unchanged.

---

# Grouping Ablation Runner: How It Works

`run_ablation_grouping_logic.py` performs the following steps:

1. Backs up the original raw input CSVs
2. Optionally backs up the current STRIKE log and saved models
3. Loads the three raw feature-group datasets
4. Applies a **stratified row subsample** using the demographic table
5. Constructs grouped ablated versions of the three raw tables
6. Writes those ablated tables to the standard STRIKE raw input filenames
7. Runs the unchanged STRIKE training script
8. Copies the resulting STRIKE log into the ablation logs folder
9. Restores the original raw input files and previous artifacts

This design allows the ablation study to reuse the exact same STRIKE training code used in the main experiment.

---

# Environment Variables for `run_ablation_grouping_logic.py`

The grouping ablation runner is controlled through environment variables.

## `GROUPING_STRATEGY`

Default:

```bash
manual
```

However, note that the currently imported grouping module is `grouping_logic_random.py`, which only supports:

```bash
random
```

So in the current code state, this should be set to:

```bash
GROUPING_STRATEGY=random
```

Using `manual` with the current import setup will fail because `grouping_logic_random.py` only accepts `strategy="random"`.

---

## `GROUPING_SEED`

Default:

```bash
42
```

Controls the random feature grouping seed.

Example:

```bash
GROUPING_SEED=42
```

---

## `ROW_SAMPLE_SIZE`

Default:

```bash
50000
```

Controls the number of rows sampled before running the grouping ablation.

Example:

```bash
ROW_SAMPLE_SIZE=50000
```

---

## `ROW_SAMPLE_SEED`

Default:

```bash
123
```

Controls the stratified row sampling seed.

Example:

```bash
ROW_SAMPLE_SEED=123
```

---

# Example Commands

## No-grouping baseline

```bash
python src/exp3_homecredit/ablation_studies/run_ablation_no_grouping.py
```

## Random mixed grouping ablation

```bash
GROUPING_STRATEGY=random GROUPING_SEED=42 ROW_SAMPLE_SIZE=50000 ROW_SAMPLE_SEED=123 \
python src/exp3_homecredit/ablation_studies/run_ablation_grouping_logic.py
```

---

# Input Data Assumptions

Both scripts expect the Home Credit data files under:

```bash
src/exp3_homecredit/data/
```

with the following filenames:

* `application_train.csv`
* `deq_features_level1.csv`
* `vintage_features_1.csv`

These files are expected to contain:

* `SK_ID_CURR`
* `TARGET` (where applicable)
* feature columns corresponding to each raw group

---

# Output Behavior

## `run_ablation_grouping_logic.py`

Produces:

* ablation logs under:

  ```bash
  src/exp3_homecredit/ablation_studies/logs/
  ```
* copied STRIKE logs archived into the same ablation logs folder
* any model outputs generated by the main STRIKE pipeline

Important: this script temporarily overwrites the raw input CSVs during execution, but restores them afterward.

---

## `run_ablation_no_grouping.py`

Produces:

* logs under:

  ```bash
  src/exp3_homecredit/ablation_studies/logs/
  ```
* saved model artifacts under:

  ```bash
  outputs/ablation_no_grouping/
  ```
* final printed and logged test AUC

---

# Important Notes

## 1. Active grouping runner currently uses `grouping_logic_random.py`

Even though `grouping_logic.py` exists and supports several grouping strategies, the current active ablation runner imports:

```python
from src.exp3_homecredit.ablation_studies.grouping_logic_random import ...
```

Therefore, the currently reproducible grouping ablation corresponds to the **random mixed grouping** setup.

---

## 2. `GROUPING_STRATEGY=manual` is not compatible with the current import path

The script default is:

```python
strategy = os.environ.get("GROUPING_STRATEGY", "manual").lower()
```

but the imported `grouping_logic_random.py` only supports:

```python
strategy == "random"
```

So users should explicitly run:

```bash
GROUPING_STRATEGY=random
```

when using the current code.

---

## 3. No code cleanup was performed here

The codebase is preserved in the form used for experimentation.
This README is intended to clarify what each script does and how to run the ablations without modifying the original experimental code.

---
