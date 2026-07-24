import pandas as pd
import numpy as np
import pickle
import os

df = pd.read_csv("outputs/cleaned_transactions.csv")
df["Date"] = pd.to_datetime(df["Date"])

print("=== BUILDING BEHAVIORAL PROFILE ===")
print(f"Total transactions: {len(df)}")

# ── FEATURE 1: Transaction Frequency ─────────────────────────
# How many transactions per month on average?
# Why: A sudden spike in transaction count is a fraud signal
monthly_counts = df.groupby(["Year","Month"]).size()
avg_monthly_freq = round(float(monthly_counts.mean()), 2)
max_monthly_freq = int(monthly_counts.max())
min_monthly_freq = int(monthly_counts.min())
print(f"\nAvg monthly transaction frequency: {avg_monthly_freq}")
print(f"Max in one month: {max_monthly_freq}")
print(f"Min in one month: {min_monthly_freq}")

# ── FEATURE 2: Spending Statistics ───────────────────────────
# Average, std deviation, min, max per transaction
# Why: A transaction far above the personal average = suspicious
avg_spend    = round(float(df["Amount"].mean()), 2)
std_spend    = round(float(df["Amount"].std()),  2)
median_spend = round(float(df["Amount"].median()), 2)
max_spend    = round(float(df["Amount"].max()), 2)

# Spending bands — what % of transactions fall in each range?
bands = {
    "Under $25":   int((df["Amount"] < 25).sum()),
    "$25–$50":     int(((df["Amount"] >= 25) & (df["Amount"] < 50)).sum()),
    "$50–$100":    int(((df["Amount"] >= 50) & (df["Amount"] < 100)).sum()),
    "$100–$200":   int(((df["Amount"] >= 100) & (df["Amount"] < 200)).sum()),
    "Over $200":   int((df["Amount"] >= 200).sum())
}
print(f"\nAvg spend: ${avg_spend}")
print(f"Std deviation: ${std_spend}")
print(f"Median: ${median_spend}")
print(f"Spending bands: {bands}")

# ── FEATURE 3: Merchant Risk Profile ─────────────────────────
# Which merchant categories does this user normally use?
# Why: A transaction at an unusual merchant type = risk signal
category_counts = df.groupby("Category").size().sort_values(ascending=False)
top_category    = category_counts.index[0]
category_pct    = (category_counts / len(df) * 100).round(2).to_dict()
print(f"\nTop merchant category: {top_category}")

# Online vs physical ratio
online_count   = len(df[df["Use Chip"] == "Online Transaction"])
physical_count = len(df[df["Use Chip"] != "Online Transaction"])
online_ratio   = round(online_count / len(df) * 100, 2)
print(f"Online transaction ratio: {online_ratio}%")

# ── FEATURE 4: Time of Day Profile ───────────────────────────
# Which hours does this customer normally transact?
# Why: A 3am transaction from someone who always shops 9am-6pm = flag
hour_counts = df.groupby("Hour").size()
peak_hour   = int(hour_counts.idxmax())

# Normal business hours (9-18) vs unusual hours
normal_hours  = df[(df["Hour"] >= 9) & (df["Hour"] <= 18)]
unusual_hours = df[(df["Hour"] < 9)  | (df["Hour"] > 18)]
normal_pct    = round(len(normal_hours) / len(df) * 100, 2)
unusual_pct   = round(len(unusual_hours) / len(df) * 100, 2)
print(f"\nPeak hour: {peak_hour}:00")
print(f"Normal hours (9am-6pm): {normal_pct}%")
print(f"Unusual hours: {unusual_pct}%")

# ── FEATURE 5: Location Consistency ──────────────────────────
# How concentrated is spending geographically?
# Why: A transaction in a new city = potential flag
city_counts   = df.groupby("Merchant City").size().sort_values(ascending=False)
top_city      = city_counts.index[0]
top_city_pct  = round(city_counts.iloc[0] / len(df) * 100, 2)
num_cities    = len(city_counts)
print(f"\nTop city: {top_city} ({top_city_pct}% of transactions)")
print(f"Total distinct cities: {num_cities}")

# ── FEATURE 6: Velocity Features ─────────────────────────────
# Velocity = rate of change in spending
# A sudden burst of transactions in a short window = card testing fraud
# We look at daily transaction counts
daily_counts      = df.groupby("Date").size()
avg_daily_txns    = round(float(daily_counts.mean()), 2)
max_daily_txns    = int(daily_counts.max())
high_velocity_days = int((daily_counts > avg_daily_txns * 2).sum())

# Also: largest single-day spend
daily_spend       = df.groupby("Date")["Amount"].sum()
max_daily_spend   = round(float(daily_spend.max()), 2)
avg_daily_spend   = round(float(daily_spend.mean()), 2)
print(f"\nAvg daily transactions: {avg_daily_txns}")
print(f"Max transactions in one day: {max_daily_txns}")
print(f"High velocity days (>2x avg): {high_velocity_days}")
print(f"Max single-day spend: ${max_daily_spend}")

# ── FEATURE 7: Fraud Pattern Profile ─────────────────────────
# What does fraud look like for this specific user?
fraud_txns    = df[df["Is Fraud?"] == "Yes"]
legit_txns    = df[df["Is Fraud?"] == "No"]
fraud_avg_amt = round(float(fraud_txns["Amount"].mean()), 2) if len(fraud_txns) > 0 else 0
legit_avg_amt = round(float(legit_txns["Amount"].mean()), 2)
fraud_online  = len(fraud_txns[fraud_txns["Use Chip"]=="Online Transaction"])
fraud_swipe   = len(fraud_txns[fraud_txns["Use Chip"]=="Swipe Transaction"])
fraud_chip    = len(fraud_txns[fraud_txns["Use Chip"]=="Chip Transaction"])

print(f"\nFraud avg amount: ${fraud_avg_amt}")
print(f"Legit avg amount: ${legit_avg_amt}")
print(f"Fraud by method — Online:{fraud_online} Swipe:{fraud_swipe} Chip:{fraud_chip}")

# ── FEATURE 8: Spending Deviation Score ──────────────────────
# For each transaction, how many std devs from the mean is it?
# We compute this as a reference for the fraud checker
def get_deviation_score(amount):
    if std_spend == 0:
        return 0
    return round(abs(amount - avg_spend) / std_spend, 2)

# ── COMPILE BEHAVIORAL PROFILE ───────────────────────────────
profile = {
    # Frequency
    "avg_monthly_freq":    avg_monthly_freq,
    "max_monthly_freq":    max_monthly_freq,
    "min_monthly_freq":    min_monthly_freq,

    # Spending
    "avg_spend":           avg_spend,
    "std_spend":           std_spend,
    "median_spend":        median_spend,
    "max_spend":           max_spend,
    "spending_bands":      bands,

    # Merchant
    "top_category":        top_category,
    "category_distribution": category_pct,
    "online_ratio":        online_ratio,

    # Time
    "peak_hour":           peak_hour,
    "normal_hours_pct":    normal_pct,
    "unusual_hours_pct":   unusual_pct,

    # Location
    "top_city":            top_city,
    "top_city_pct":        top_city_pct,
    "num_distinct_cities": num_cities,

    # Velocity
    "avg_daily_txns":      avg_daily_txns,
    "max_daily_txns":      max_daily_txns,
    "high_velocity_days":  high_velocity_days,
    "max_daily_spend":     max_daily_spend,
    "avg_daily_spend":     avg_daily_spend,

    # Fraud pattern
    "fraud_avg_amount":    fraud_avg_amt,
    "legit_avg_amount":    legit_avg_amt,
    "fraud_by_method": {
        "Online": fraud_online,
        "Swipe":  fraud_swipe,
        "Chip":   fraud_chip
    },

    # Thresholds (used by fraud checker)
    "high_spend_threshold":   round(avg_spend + 2 * std_spend, 2),
    "extreme_spend_threshold": round(avg_spend + 3 * std_spend, 2),
}

os.makedirs("model", exist_ok=True)
with open("model/behavioral_profile.pkl", "wb") as f:
    pickle.dump(profile, f)

print("\n=== BEHAVIORAL PROFILE SUMMARY ===")
print(f"Normal spend range:     $0 – ${profile['high_spend_threshold']}")
print(f"High spend threshold:   ${profile['high_spend_threshold']}")
print(f"Extreme spend threshold:${profile['extreme_spend_threshold']}")
print(f"Normal hours:           {normal_pct}% of transactions")
print(f"Online ratio:           {online_ratio}%")
print(f"Avg daily transactions: {avg_daily_txns}")
print("\nBehavioral profile saved to model/behavioral_profile.pkl")