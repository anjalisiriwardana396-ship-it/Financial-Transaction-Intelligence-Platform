import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import shap

df = pd.read_csv("outputs/cleaned_transactions.csv")

with open("model/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/fraud_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("model/fraud_features.pkl", "rb") as f:
    features = pickle.load(f)
with open("model/fraud_best_name.pkl", "rb") as f:
    best_name = pickle.load(f)

print(f"Explaining: {best_name}")

# Prepare data the same way as training
text_cols = ["Use Chip", "Merchant State", "Errors?", "DayOfWeek"]
df_model = df[features].copy()
for col in text_cols:
    df_model[col] = encoders[col].transform(df_model[col].astype(str))

sample = df_model.sample(200, random_state=42)
print("Running SHAP on 200 sample transactions...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# ── FIX: handle both formats SHAP can return ────────────────
# Older SHAP returns a list [class0_values, class1_values]
# Newer SHAP returns a 3D array of shape (samples, features, classes)
# We want class 1 (fraud) values in all cases
if isinstance(shap_values, list):
    # Old format: shap_values is a list, index 1 = fraud class
    sv = shap_values[1]
    print("SHAP format: list (older version)")
else:
    # New format: shap_values is a 3D numpy array
    # shape = (200 samples, 8 features, 2 classes)
    # [:, :, 1] means: all samples, all features, class 1 (fraud)
    sv = shap_values[:, :, 1]
    print("SHAP format: 3D array (newer version)")

print(f"SHAP values shape: {sv.shape}")
print(f"Sample shape: {sample.shape}")
print("SHAP values calculated successfully")

# ── CHART 1: Bar chart — overall feature importance ──────────
plt.figure()
shap.summary_plot(
    sv,
    sample,
    feature_names=features,
    plot_type="bar",
    show=False
)
plt.title("Feature Importance — What drives fraud prediction (SHAP)")
plt.tight_layout()
plt.savefig("outputs/09_shap_bar.png", dpi=100, bbox_inches="tight")
plt.close()
print("Bar chart saved to outputs/09_shap_bar.png")

# ── CHART 2: Dot plot — direction of impact ──────────────────
plt.figure()
shap.summary_plot(
    sv,
    sample,
    feature_names=features,
    plot_type="dot",
    show=False
)
plt.title("SHAP Detail — Direction of each feature's impact on fraud")
plt.tight_layout()
plt.savefig("outputs/10_shap_dot.png", dpi=100, bbox_inches="tight")
plt.close()
print("Dot plot saved to outputs/10_shap_dot.png")

# ── Plain English summary ────────────────────────────────────
print("\n=== SHAP FINDINGS ===")
mean_shap = np.abs(sv).mean(axis=0)
shap_df = pd.DataFrame({
    "Feature":    features,
    "Importance": mean_shap.round(4)
}).sort_values("Importance", ascending=False)

print(shap_df.to_string(index=False))
print(f"\nMost influential feature:  {shap_df.iloc[0]['Feature']}")
print(f"Least influential feature: {shap_df.iloc[-1]['Feature']}")
print("\nCheck outputs/ for 09_shap_bar.png and 10_shap_dot.png")