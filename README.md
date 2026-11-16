# Customer Churn Prediction

XGBoost model that predicts whether a customer will churn, built on tabular
customer data (tenure, spend, support tickets, contract type, age, payment failures).

## Problem
Churn is a binary classification task on an imbalanced dataset (~25% churn).
The goal is a model that actually finds churning customers, not one that
just predicts "no churn" for everyone.

## Approach
- 6 customer features, 80/20 stratified train/test split
- XGBoost with `scale_pos_weight` set to the neg/pos ratio to handle the imbalance
- Evaluation with precision / recall / F1, not just accuracy (accuracy is misleading on imbalanced data)
- Data leak experiment: added a synthetic feature (`days_since_last_payment`)
  correlated with the label, and compared the model with and without it

## Results

| Model | Accuracy | Churn Recall | Churn F1 |
|---|---|---|---|
| Honest (no leak) | 0.58 | 0.53 | 0.39 |
| With leaked feature | 0.76 | 0.37 | 0.43 |

The leak raises accuracy by ~18 points but the model still misses most churners —
inflated metrics don't mean a usable model. The honest model catches about half of
actual churners, a realistic baseline for this feature set.

## Files
- `churn_model.py` — full pipeline: data generation, training, leak experiment, evaluation
- `requirements.txt` — dependencies
- `confusion_matrix.png` — confusion matrix of the honest model
