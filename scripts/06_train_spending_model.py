import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle
import os

df = pd.read_csv("outputs/cleaned_transactions.csv")

# ── STEP 1: Collapse to monthly totals ──────────────────────
# We want to predict one number per month (total spending) not predict individual transactions
monthly = df.groupby(["Year", "Month"])["Amount"].sum().reset_index()

monthly = monthly.sort_values(["Year", "Month"]).reset_index(drop=True)
monthly["MonthIndex"] = range(len(monthly))
# MonthIndex = 0, 1, 2, 3... just a simple counter
# Month 0 = Sept 2002, Month 1 = Oct 2002, etc.

print("Monthly data sample:")
print(monthly.head())
print(f"\nTotal months in dataset: {len(monthly)}")

# ── STEP 2: Create lag features ──────────────────────────────
# The model needs to know recent history to predict next month
# Without this, it only knows "it's month 50" which tells it almost nothing

# Lag1 = last month's total spending
monthly["Lag1"] = monthly["Amount"].shift(1)
# shift(1) moves all values down by 1 row
# so row 5's Lag1 = row 4's Amount = last month's spending

# Lag2 = two months ago
monthly["Lag2"] = monthly["Amount"].shift(2)

# Lag3 = three months ago
monthly["Lag3"] = monthly["Amount"].shift(3)

# RollingMean3 = average of last 3 months
# Smooths out one-off spikes, gives trend direction
monthly["RollingMean3"] = monthly["Amount"].shift(1).rolling(3).mean()
# shift(1) first so we don't include the current month in its own average
# rolling(3).mean() = sliding window average of 3 values

# Month number = captures patterns like "December is always high"
monthly["Month_num"] = monthly["Month"]

# Year = captures long term growth over 18 years
monthly["Year_num"] = monthly["Year"]

# Festive period as a number
# Models need numbers, not text
def get_festive_num(m):
    if m == 4:           return 4   # Avurudu
    elif m == 12:        return 3   # Christmas
    elif m == 1:         return 2   # New Year
    elif m == 2:         return 1   # Valentine
    else:                return 0   # Normal

monthly["Festive_num"] = monthly["Month"].apply(get_festive_num)

# ── STEP 3: Drop first 3 rows ────────────────────────────────
# Why: Lag1/Lag2/Lag3 for the first 3 months are NaN (no history yet)
# The model cannot handle NaN values
monthly_clean = monthly.dropna().reset_index(drop=True)
print(f"Usable months after removing NaN lags: {len(monthly_clean)}")

# ── STEP 4: Define features and target ──────────────────────
features = ["Lag1", "Lag2", "Lag3", "RollingMean3",
            "Month_num", "Year_num", "Festive_num", "MonthIndex"]

X = monthly_clean[features]   # inputs
y = monthly_clean["Amount"]   # what we want to predict

# ── STEP 5: Split into train and test ────────────────────────
# We train on 80% of months, test on the remaining 20%
# The model must NEVER see test data during training
# For time series data we split by position, not randomly
# (We can't train on 2015 data to predict 2010 — time only goes forward)
split = int(len(monthly_clean) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nTraining on {len(X_train)} months")
print(f"Testing on  {len(X_test)} months")

# ── STEP 6: Train the model ──────────────────────────────────
# LinearRegression finds the best straight-line relationship
# between your features and the target
# It figures out: Amount ≈ (w1 × Lag1) + (w2 × Lag2) + ... + constant
# where w1, w2 etc. are weights it learns automatically
model = LinearRegression()
model.fit(X_train, y_train)
# .fit() = the actual learning step

# ── STEP 7: Evaluate on TEST set only ───────────────────────
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
# MAE = Mean Absolute Error
# Average of |actual - predicted| across all test months
# Lower MAE = better model
print(f"\nModel MAE (Mean Absolute Error): ${mae:.2f}")
print(f"This means predictions are off by about ${mae:.2f} on average")

# Show actual vs predicted for last 5 test months
comparison = pd.DataFrame({
    "Actual":    y_test.values[-5:],
    "Predicted": preds[-5:].round(2)
})
print("\nActual vs Predicted (last 5 test months):")
print(comparison.to_string(index=False))

# ── STEP 8: Predict next 6 months ───────────────────────────
# This is the useful output — what does next month look like?
# We roll forward: each prediction becomes the next row's lag input
recent = list(monthly_clean["Amount"].tail(3).values)
predictions_future = []

for i in range(6):
    lag1     = recent[-1]
    lag2     = recent[-2]
    lag3     = recent[-3]
    rolling3 = np.mean(recent[-3:])

    future_month = ((monthly_clean.iloc[-1]["Month_num"] + i) % 12) + 1
    future_year  = monthly_clean.iloc[-1]["Year_num"] + (i // 12)
    festive      = get_festive_num(int(future_month))
    future_idx   = monthly_clean.iloc[-1]["MonthIndex"] + i + 1

    row  = [[lag1, lag2, lag3, rolling3,
             future_month, future_year, festive, future_idx]]
    pred = model.predict(row)[0]
    predictions_future.append(pred)
    recent.append(pred)

print("\n=== NEXT 6 MONTHS PREDICTIONS ===")
for i, pred in enumerate(predictions_future):
    print(f"  Future month {i+1}: ${pred:,.2f}")

# ── STEP 9: Save model and metadata ─────────────────────────
#  Flask app needs to load this model to make predictions
# pickle saves a Python object to a file so it can be reloaded later
os.makedirs("model", exist_ok=True)

with open("model/spending_model.pkl", "wb") as f:
    pickle.dump(model, f)
# "wb" = write binary (pickle files are binary, not text)

last_known = {
    "recent_amounts": list(monthly_clean["Amount"].tail(3).values),
    "last_month_num": int(monthly_clean.iloc[-1]["Month_num"]),
    "last_year":      int(monthly_clean.iloc[-1]["Year_num"]),
    "last_idx":       int(monthly_clean.iloc[-1]["MonthIndex"]),
    "mae":            round(mae, 2),
    "features":       features
}
with open("model/month_info.pkl", "wb") as f:
    pickle.dump(last_known, f)

print("\nSpending model saved to model/spending_model.pkl")
print("Month info saved to model/month_info.pkl")    