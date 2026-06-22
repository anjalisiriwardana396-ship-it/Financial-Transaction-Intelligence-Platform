import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("outputs/cleaned_transactions.csv")
df["Date"] = pd.to_datetime(df["Date"])

# sns.set_theme() makes all charts look cleaner than default matplotlib style
sns.set_theme(style="whitegrid")

# ── CHART 1: Yearly Spending Trend ──────────────────────────
# Shows how spending changed across 18 years
yearly = df.groupby("Year")["Amount"].sum()

plt.figure(figsize=(12, 5))


yearly.plot(kind="line", marker="o", color="steelblue", linewidth=2.5)


plt.title("Yearly Spending Trend (2002–2020)")
plt.xlabel("Year")
plt.ylabel("Total Amount ($)")
plt.tight_layout()


plt.savefig("outputs/01_yearly_trend.png", dpi=100)

plt.close()

print("Chart 1 saved")

# ── CHART 2: Monthly Spending Heatmap ───────────────────────
#Shows which month of which year had highest spending

pivot = df.pivot_table(values="Amount", index="Year",
                        columns="Month", aggfunc="sum")

plt.figure(figsize=(14, 8))
sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.5)

plt.title("Monthly Spending Heatmap (Yellow=Low, Red=High)")
plt.tight_layout()
plt.savefig("outputs/02_monthly_heatmap.png", dpi=100)
plt.close()
print("Chart 2 saved")

# ── CHART 3: Spending by Category ───────────────────────────
#Shows where the most money actually goes
cat = df.groupby("Category")["Amount"].sum().sort_values()


plt.figure(figsize=(10, 6))
cat.plot(kind="barh", color="teal")

plt.title("Total Spending by Category")
plt.xlabel("Amount ($)")
plt.tight_layout()
plt.savefig("outputs/03_category.png", dpi=100)
plt.close()
print("Chart 3 saved")

# ── CHART 4: Fraud by Payment Method ────────────────────────
#  Visually proves online transactions are the riskiest
fraud_by_method = df.groupby("Use Chip").apply(
    lambda x: (x["Is Fraud?"] == "Yes").sum()
).reset_index()
fraud_by_method.columns = ["Method", "FraudCount"]
# lambda x = a mini function applied to each group
# (x["Is Fraud?"] == "Yes").sum() = count how many rows are fraud

plt.figure(figsize=(8, 5))
colors = ["steelblue", "coral", "mediumpurple"]
plt.bar(fraud_by_method["Method"], fraud_by_method["FraudCount"], color=colors)
plt.title("Fraud Cases by Payment Method")
plt.ylabel("Number of Fraud Cases")
plt.xticks(rotation=15)
# rotation=15 tilts the x-axis labels slightly so they don't overlap
plt.tight_layout()
plt.savefig("outputs/04_fraud_by_method.png", dpi=100)
plt.close()
print("Chart 4 saved")

# ── CHART 5: Spending by Day of Week ────────────────────────
# Shows if weekends or weekdays drive more spending
dow_order = ["Monday","Tuesday","Wednesday","Thursday",
             "Friday","Saturday","Sunday"]
dow = df.groupby("DayOfWeek")["Amount"].sum().reindex(dow_order)


plt.figure(figsize=(10, 5))
dow.plot(kind="bar", color="coral")
plt.title("Total Spending by Day of Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/05_day_of_week.png", dpi=100)
plt.close()
print("Chart 5 saved")

# ── CHART 6: Transaction Amount Distribution ─────────────────
#  Shows if most transactions are small or large
# A histogram puts amounts into buckets and counts how many fall in each
plt.figure(figsize=(10, 5))
df[df["Amount"] > 0]["Amount"].hist(bins=50,
    color="steelblue", edgecolor="white")
# bins=50 = split the range into 50 buckets
# df[df["Amount"] > 0] = exclude any refunds/zero amounts
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount ($)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("outputs/06_amount_dist.png", dpi=100)
plt.close()
print("Chart 6 saved")

# ── CHART 7: Spending by Festive Period ──────────────────────
# Shows which festive period drives the most spending
festive = df.groupby("FestivePeriod")["Amount"].sum()

plt.figure(figsize=(8, 5))
festive.sort_values().plot(kind="bar", color="mediumpurple")
plt.title("Total Spending by Festive Period")
plt.xlabel("Festive Period")
plt.ylabel("Total Amount ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/07_festive.png", dpi=100)
plt.close()
print("Chart 7 saved")

print("\nAll 7 charts saved to outputs/ folder")
