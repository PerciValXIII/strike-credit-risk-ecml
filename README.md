# STRIKE: Additive Feature-Group-Aware Stacking Framework for Credit Default Prediction

This repository contains the official implementation of **STRIKE**, introduced in our ECML PKDD 2026 ADS Track submission:  
**"STRIKE: Additive Feature-Group-Aware Stacking Framework for Credit Default Prediction"**

STRIKE is a modular stacking framework that enhances predictive performance by:
- Isolating semantically coherent **feature groups**,
- Training specialized **base models** on each group,
- Aggregating their predictions using a **logistic regression meta-learner**.

We provide scripts to reproduce results on three benchmark datasets:
- Polish Company Bankruptcy Dataset (UCI)
- LendingClub Peer-to-Peer Loan Dataset (Kaggle)
- Home Credit Default Risk Dataset (Kaggle)

---

## Setup Instructions

### Clone & Install Dependencies

```bash
git clone https://github.com/PerciValXIII/strike-credit-risk.git
cd strike-credit-risk
conda create -n strike python=3.10 -y
conda activate strike
pip install -r requirements.txt
```

---

## Reproducing Experiments

Each experiment is located under `src/` in its own subdirectory:

- `exp1_polish/`
- `exp2_lendingclub/`
- `exp3_homecredit/`

### Running STRIKE

Please refer to the `README.md` inside each experiment folder for exact commands.

Each script performs the following steps:
- Loads and preprocesses dataset 
- Constructs isolated feature groups
- Trains multiple base models with out-of-fold predictions
- Builds a meta-dataset
- Trains a logistic regression meta-model

---

## 📁 Project Structure

```
├── generic_notebooks/       # Jupyter notebooks for exploratory work
├── outputs/
│   ├── logs/                # Logs for training and evaluation
├── src/
│   ├── exp1_polish/
│   ├── exp2_lendingclub/
│   ├── exp3_homecredit/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Evaluation Protocol

- **Primary Metric**: AUC-ROC
- Dataset Split: 70/30 train-test
- 5-Fold CV used for base models
- Meta-model trained on OOF predictions
- All experiments are reproducible from raw CSV to final predictions

---

## Reference Datasets

- [Polish Bankruptcy Dataset (UCI)](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
- [LendingClub Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk)

---

## Citation

If you use STRIKE in your work, please cite:

```bibtex
@misc{maiti2024strike,
  author       = {Swattik Maiti},
  title        = {STRIKE: Stacking via Targeted Representations of Isolated Knowledge Extractors},
  year         = {2025},
  url          = {https://github.com/PerciValXIII/strike-credit-risk},
}
```

---

## License

This project is licensed under the MIT License.

---

## Contributors 
Swattik Maiti, Ritik Pratap Singh, Fardina Alam
