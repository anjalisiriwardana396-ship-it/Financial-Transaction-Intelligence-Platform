import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("outputs/cleaned_transactions.csv")

# ── METHOD 1: IQR (Interquartile Range) ─────────────────────
# Standard method to find statistically unusual values

Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: ${Q1:.2f}")
print(f"Q3: ${Q3:.2f}")
print(f"IQR: ${IQR:.2f}")
print(f"Normal range: ${lower_bound:.2f} to ${upper_bound:.2f}")

outliers_iqr = df[(df["Amount"] < lower_bound) | (df["Amount"] > upper_bound)]
print(f"\nOutliers found(IQR method): {len(outliers_iqr)}")
print("\nTop 10 largest outlier transactions:")
print(outliers_iqr[["Date", "Amount", "Merchant City",
                     "Category", "Is Fraud?", "Use Chip"]]
      .sort_values("Amount", ascending=False)
      .head(10).to_string())

# ── METHOD 2: Z-Score ────────────────────────────────────────
df["z_score"] = np.abs(stats.zscore(df["Amount"]))
zscore_outliers = df[df["z_score"] > 3]

print(f"\nZ-score outliers (z > 3): {len(zscore_outliers)}")
print("\nTop 10 by z-score:")
print(zscore_outliers[["Date", "Amount", "z_score",
                        "Is Fraud?", "Use Chip"]]
      .sort_values("z_score", ascending=False)
      .head(10).to_string())

# ── REFUNDS ──────────────────────────────────────────────────
# Why: Negative amounts mean money came back (refund or reversal)
# These are not fraud but worth knowing about
negatives = df[df["Amount"] < 0]
print(f"\nRefund/reversal transactions: {len(negatives)}")
if len(negatives) > 0:
    print(negatives[["Date", "Amount", "Category", "Use Chip"]].head())

# ── HOUR 6 INVESTIGATION ─────────────────────────────────────
# Why: In Phase 3 we saw $875k spent at Hour 6 which looked suspicious
# Let's investigate what's actually happening at that hour
print("\n--- Hour 6 Investigation ---")
hour6 = df[df["Hour"] == 6]
print(f"Total transactions at hour 6: {len(hour6)}")
print(f"Total amount at hour 6: ${hour6['Amount'].sum():,.2f}")
print(f"Average amount at hour 6: ${hour6['Amount'].mean():.2f}")
print(f"Most common category at hour 6:")
print(hour6["Category"].value_counts().head(5))
# This tells us if hour 6 is genuinely suspicious or just a data quirk

# ── CHART: Boxplot ───────────────────────────────────────────
# A boxplot visually shows outliers as individual dots

plt.figure(figsize=(10, 5))
plt.boxplot(df["Amount"], vert=False, patch_artist=True,
            boxprops=dict(facecolor="lightblue"))
plt.title("Transaction Amounts — dots beyond whiskers are outliers")
plt.xlabel("Amount ($)")
plt.tight_layout()
plt.savefig("outputs/08_outliers_boxplot.png", dpi=100)
plt.close()
print("\nBoxplot saved to outputs/08_outliers_boxplot.png")

# ── SAVE OUTLIER REPORT ──────────────────────────────────────
outliers_iqr.to_csv("outputs/outliers.csv", index=False)
print("Outlier report saved to outputs/outliers.csv")
print(f"\nSUMMARY: {len(outliers_iqr)} outliers out of {len(df)} transactions")
print(f"That is {len(outliers_iqr)/len(df)*100:.1f}% of all transactions")   