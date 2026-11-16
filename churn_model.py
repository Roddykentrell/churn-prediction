import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# 1. Synthetic customer data (realistic features)
np.random.seed(42)
n = 5000
data = {
    "tenure_months": np.random.randint(1, 60, n),
    "monthly_spend": np.random.normal(50, 20, n).clip(5),
    "support_tickets": np.random.poisson(2, n),
    "contract_years": np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2]),
    "age": np.random.randint(18, 70, n),
    "payment_failures": np.random.poisson(0.5, n),
}
df = pd.DataFrame(data)

# Churn label: more likely with short tenure, low spend, many tickets/failures
p = (0.05 + 0.002 * df.support_tickets + 0.001 * df.payment_failures
     - 0.003 * df.tenure_months - 0.001 * df.monthly_spend)
df["churned"] = (np.random.rand(n) < p.clip(0.01, 0.6)).astype(int)
print(f"Class balance: {df.churned.mean():.1%} churned")

# 2. Data leak demo: a feature that "predicts" the outcome
df["days_since_last_payment"] = np.where(df.churned == 1,
    np.random.randint(30, 90, n), np.random.randint(0, 10, n))

# 3. Features (without the leak)
features = ["tenure_months", "monthly_spend", "support_tickets",
            "contract_years", "age", "payment_failures"]
X = df[features]
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

# 4. Model — handle imbalance with scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                      scale_pos_weight=neg / pos, eval_metric="logloss")
model.fit(X_train, y_train)

# 5. Evaluate
print("\n--- With leak removed (honest model) ---")
print(classification_report(y_test, model.predict(X_test)))

leaky = X_train.assign(days_since_last_payment=df["days_since_last_payment"].iloc[:len(X_train)])
leaky_test = X_test.assign(days_since_last_payment=df["days_since_last_payment"].iloc[len(X_train):])
leak_model = XGBClassifier(n_estimators=200, max_depth=4)
leak_model.fit(leaky, y_train)
print("--- With leak (inflated, looks too good) ---")
print(classification_report(y_test, leak_model.predict(leaky_test)))

# 6. Save confusion matrix chart
cm = confusion_matrix(y_test, model.predict(X_test))
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Churn Model — Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=100)
print("\nSaved confusion_matrix.png")
