import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ─────────────────────────────────────────
#  STEP 1 — LOAD DATASET
# ─────────────────────────────────────────

print("=" * 50)
print("  ATM FRAUD DETECTION — MODEL TRAINING")
print("=" * 50)

df = pd.read_csv("transactions.csv")

print(f"\n📂 Dataset loaded")
print(f"   Total transactions : {len(df)}")
print(f"   Fraud              : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"   Legitimate         : {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")

# ─────────────────────────────────────────
#  STEP 2 — PREPARE FEATURES
# ─────────────────────────────────────────

FEATURES = [
    "amount",
    "hour_of_day",
    "pin_attempts",
    "is_new_location",
    "txn_frequency_10min",
    "amount_vs_avg"
]

X = df[FEATURES]
y = df["label"]

# ─────────────────────────────────────────
#  STEP 3 — TRAIN / TEST SPLIT
#  80% training, 20% testing
#  stratify=y keeps fraud ratio same in both splits
# ─────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n📊 Train/Test Split")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")

# ─────────────────────────────────────────
#  STEP 4 — TRAIN BASELINE MODEL
#  No optimization — just a plain Random Forest
# ─────────────────────────────────────────

print("\n⏳ Training baseline model (no class weights)...")

model_baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)

print("✅ Baseline model trained")

# ─────────────────────────────────────────
#  STEP 5 — TRAIN WEIGHTED MODEL
#  Missing fraud (false negative) = penalized 15x
#  This is your core ML optimization
# ─────────────────────────────────────────

print("\n⏳ Training weighted model (fraud penalized 15x)...")

model_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight={0: 1, 1: 15},
    random_state=42
)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)

print("✅ Weighted model trained")

# ─────────────────────────────────────────
#  STEP 6 — EVALUATE BOTH MODELS
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("  BASELINE MODEL (no class weights)")
print("=" * 50)
print(classification_report(y_test, y_pred_baseline,
      target_names=["Legitimate", "Fraud"]))

print("=" * 50)
print("  WEIGHTED MODEL (fraud penalized 15x)")
print("=" * 50)
print(classification_report(y_test, y_pred_weighted,
      target_names=["Legitimate", "Fraud"]))

# ─────────────────────────────────────────
#  STEP 7 — CONFUSION MATRIX COMPARISON
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight="bold")

cm_baseline = confusion_matrix(y_test, y_pred_baseline)
cm_weighted  = confusion_matrix(y_test, y_pred_weighted)

for ax, cm, title in zip(
    axes,
    [cm_baseline, cm_weighted],
    ["Baseline (no weights)", "Weighted (fraud = 15x)"]
):
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        xticklabels=["Predicted Legit", "Predicted Fraud"],
        yticklabels=["Actual Legit", "Actual Fraud"],
        ax=ax
    )
    ax.set_title(title, fontweight="bold")
    fn = cm[1][0]
    ax.set_xlabel(f"False Negatives (missed fraud): {fn}", color="red", fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png", dpi=150)
print("\n📊 Confusion matrix saved → confusion_matrix_comparison.png")

# ─────────────────────────────────────────
#  STEP 8 — FEATURE IMPORTANCE
# ─────────────────────────────────────────

importances = model_weighted.feature_importances_
feat_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n🔍 Feature Importances (weighted model):")
for _, row in feat_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"   {row['Feature']:<22} {bar} {row['Importance']:.4f}")

# ─────────────────────────────────────────
#  STEP 9 — ROC AUC SCORE
# ─────────────────────────────────────────

auc_baseline = roc_auc_score(y_test, model_baseline.predict_proba(X_test)[:, 1])
auc_weighted  = roc_auc_score(y_test, model_weighted.predict_proba(X_test)[:, 1])

print(f"\n📈 ROC AUC Score")
print(f"   Baseline : {auc_baseline:.4f}")
print(f"   Weighted : {auc_weighted:.4f}")

# ─────────────────────────────────────────
#  STEP 10 — SAVE THE FINAL MODEL
# ─────────────────────────────────────────

with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model_weighted, f)

print(f"\n💾 Model saved → fraud_model.pkl")
print("\n✅ Training complete. You're ready for serial_handler.py")