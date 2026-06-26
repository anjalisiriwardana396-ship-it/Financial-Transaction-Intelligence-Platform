import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("outputs/cleaned_transactions.csv")

print("=== CLASS IMBALANCE CHECK ===")
print(df["Is Fraud?"].value_counts())
print("\nProblem: only 27 fraud out of 19,963 transactions")
print("A model that always says No Fraud = 99.8% accuracy but useless")
print("Fix: split first, then SMOTE only on training data\n")

# ── STEP 1: Select features ──────────────────────────────────
# Why these features: they are all things you would know at the
# moment a transaction happens — amount, time, method, location
features = ["Amount", "Hour", "MCC", "Use Chip",
            "Merchant State", "Errors?", "Month", "DayOfWeek"]

# ── STEP 2: Encode text columns to numbers ───────────────────
# Why: Every ML model requires numbers as input, not text
# LabelEncoder converts each unique text value to a number
# e.g. "Swipe Transaction"→0, "Chip Transaction"→1, "Online Transaction"→2
text_columns = ["Use Chip", "Merchant State", "Errors?", "DayOfWeek"]
encoders = {}
df_model = df[features + ["Is Fraud?"]].copy()

for col in text_columns:
    encoders[col] = LabelEncoder()
    df_model[col] = encoders[col].fit_transform(df_model[col].astype(str))
# We save each encoder in a dictionary
# Why: Flask app needs to encode new transactions the same way

df_model["Is Fraud?"] = (df_model["Is Fraud?"] == "Yes").astype(int)
# Convert "Yes"/"No" to 1/0
# (df["Is Fraud?"] == "Yes") gives True/False
# .astype(int) converts True→1, False→0

X = df_model[features]
y = df_model["Is Fraud?"]

print(f"Total fraud cases: {y.sum()}")
print(f"Total legitimate: {(y==0).sum()}")

# ── STEP 3: Split FIRST on original imbalanced data ──────────
# WHY ORDER MATTERS:
#
# WRONG order → data leakage:
#   SMOTE → split → train → test
#   Fake fraud rows exist in BOTH train AND test sets
#   Model looks better than it really is
#
# CORRECT order → honest results:
#   split → SMOTE on train only → train → test on real data only
#
# stratify=y means: keep the same fraud ratio in both train and test
# Without this, all 27 fraud cases might end up in train and test has none
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain — Fraud: {y_train.sum()}, Legit: {(y_train==0).sum()}")
print(f"Test  — Fraud: {y_test.sum()},  Legit: {(y_test==0).sum()}")

# ── STEP 4: Apply SMOTE only to training data ────────────────
# Why: We create synthetic fraud rows so the model sees enough
# fraud examples to learn what fraud looks like
# SMOTE works by: finding a real fraud row, finding its nearest
# neighbors, and creating new rows in between them
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# fit_resample creates new synthetic rows until classes are balanced

print(f"\nAfter SMOTE — Train Fraud: {y_train_res.sum()}, "
      f"Train Legit: {(y_train_res==0).sum()}")
print(f"Test set unchanged — Fraud: {y_test.sum()}, "
      f"Legit: {(y_test==0).sum()}")

# ── STEP 5: Train and compare 3 models ──────────────────────
# Why compare 3: Shows you evaluated options, not just picked one blindly
# Logistic Regression = simple, fast, interpretable baseline
# Random Forest = many decision trees voting together, handles complexity
# XGBoost = powerful boosting model, often wins competitions
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric="logloss", verbosity=0)
}

results = []
trained_models = {}

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, clf in models.items():
    # Train on SMOTE-balanced training data
    clf.fit(X_train_res, y_train_res)

    # Predict on REAL test data (no synthetic rows)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    # predict_proba gives probability of fraud (column 1 = fraud probability)

    report = classification_report(y_test, y_pred,
                target_names=["Legit", "Fraud"],
                output_dict=True)

    precision = report["Fraud"]["precision"]
    recall    = report["Fraud"]["recall"]
    f1        = report["Fraud"]["f1-score"]
    roc_auc   = roc_auc_score(y_test, y_prob)

    # What these metrics mean:
    # Precision = of all transactions flagged as fraud, how many were real fraud?
    # Recall    = of all real fraud cases, how many did the model catch?
    #             (This is the most important one — missing fraud is costly)
    # F1        = balance between precision and recall
    # ROC-AUC   = overall ability to separate fraud from legit (1.0 = perfect)

    results.append({
        "Model":     name,
        "Precision": round(precision, 3),
        "Recall":    round(recall, 3),
        "F1-Score":  round(f1, 3),
        "ROC-AUC":   round(roc_auc, 4)
    })
    trained_models[name] = clf

    print(f"\n--- {name} ---")
    print(f"  Precision : {precision:.3f}  (of flagged fraud, this % was real)")
    print(f"  Recall    : {recall:.3f}  (of real fraud, this % was caught)")
    print(f"  F1-Score  : {f1:.3f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")

# ── STEP 6: Comparison table ─────────────────────────────────
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("FINAL COMPARISON TABLE")
print("=" * 60)
print(results_df.to_string(index=False))
results_df.to_csv("outputs/model_comparison.csv", index=False)

# ── STEP 7: Pick best model by ROC-AUC ──────────────────────
best_name = results_df.loc[results_df["ROC-AUC"].idxmax(), "Model"]
best_model = trained_models[best_name]
print(f"\nBest model selected: {best_name}")

# ── STEP 8: Feature importance ───────────────────────────────
# Why: Shows which transaction details matter most for fraud detection
rf_model = trained_models["Random Forest"]
importance = pd.DataFrame({
    "Feature":    features,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== FEATURE IMPORTANCE (Random Forest) ===")
print("Higher = more useful for detecting fraud")
print(importance.to_string(index=False))

# ── STEP 9: Save everything ──────────────────────────────────
with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("model/fraud_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open("model/fraud_features.pkl", "wb") as f:
    pickle.dump(features, f)
with open("model/fraud_best_name.pkl", "wb") as f:
    pickle.dump(best_name, f)

print(f"\nBest model ({best_name}) saved to model/fraud_model.pkl")
print("Encoders saved to model/fraud_encoders.pkl")