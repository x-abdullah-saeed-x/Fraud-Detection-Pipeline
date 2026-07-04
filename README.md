# Fraud Detection Pipeline

A machine learning pipeline for detecting fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). The pipeline handles severe class imbalance with SMOTE and compares Logistic Regression against Random Forest, tuned via GridSearchCV and evaluated with ROC-AUC.

## Problem

Credit card fraud detection is a highly imbalanced binary classification problem — fraudulent transactions make up roughly 0.17% of the dataset. A naive classifier can score >99% accuracy while catching zero fraud. This project addresses that with resampling (SMOTE) and threshold-independent evaluation (ROC-AUC).

## Dataset

- **Source:** `creditcard.csv` (Kaggle Credit Card Fraud Detection dataset)
- **Features:** `V1`–`V28` (PCA-transformed, anonymized), `Time`, `Amount`
- **Target:** `Class` (0 = legitimate, 1 = fraud)
- `Time` and `Amount` are dropped prior to modeling; only the anonymized PCA components are used as features.

## Pipeline Overview

```
Raw Data → Drop NA → Train/Test Split (stratified, 80/20)
         → [Scale (LR only)] → SMOTE (train fold only, via imblearn Pipeline)
         → Classifier → GridSearchCV (5-fold CV, scoring=roc_auc)
         → Evaluation (classification_report, ROC curve, AUC)
```

### Why `imblearn.Pipeline`?

SMOTE is applied **inside** the pipeline rather than on the full training set upfront. This ensures resampling happens fresh within each cross-validation fold, preventing synthetic samples generated from the training fold from leaking into the validation fold — a common source of inflated CV scores in imbalanced classification.

## Models

| Model | Preprocessing | Tuned Hyperparameters |
|---|---|---|
| Logistic Regression | `StandardScaler` → `SMOTE` | `C`: [0.1, 1, 10]; `solver`: [lbfgs, liblinear] |
| Random Forest | `SMOTE` | `n_estimators`: [100, 300]; `max_depth`: [5, None]; `min_samples_split`: [2, 5] |

Both models are tuned with `GridSearchCV` (5-fold CV, `scoring='roc_auc'`, `n_jobs=-1`).

> Random Forest doesn't require feature scaling, so `StandardScaler` is omitted from its pipeline.

## Results

| Model | Best CV ROC-AUC | Best Params |
|---|---|---|---|
| Logistic Regression | 0.9843 | {'C': 0.1, 'solver': 'lbfgs'} |
| Random Forest | 0.9959 | {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 100} |

See `FD Report.pdf` for full classification reports, ROC curves, and discussion.

## Tech Stack

- **Data/ML:** pandas, numpy, scikit-learn, imbalanced-learn (SMOTE)
- **Visualization:** matplotlib

## Project Structure

```
.
├── fraud_detection.py     # Main pipeline script
├── creditcard.csv         # Dataset (not included — download from Kaggle)
├── README.md
└── PROJECT_REPORT.md
```

## Setup & Usage

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib

# Place creditcard.csv in the project root, then:
python fraud_detection.py
```

## Future Improvements

- Add precision-recall curves (more informative than ROC for extreme imbalance)
- Cost-sensitive evaluation (false negatives ≫ false positives in cost for fraud)
- Try XGBoost/LightGBM with `scale_pos_weight` as a SMOTE alternative
- Persist the winning pipeline with `joblib` for deployment
- Feature importance / SHAP analysis on the Random Forest model

## License

MIT