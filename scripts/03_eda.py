import pandas as pd

df = pd.read_csv("outputs/cleaned_transactions.csv  ")
df["Date"] = pd.to_datetime(df["Date"])

print("="*50)
print("Business Summary")
print("="*50)
print(f"Total Transactions: {len(df):,}")
print(f"Total spent       : ${df['Amount'].sum():,.2f}")
print(f"Average per txn   : ${df['Amount'].mean():.2f}")
print(f"Largest single txn: ${df['Amount'].max():.2f}")
print(f"Date range        : {df['Date'].min().date()} to {df['Date'].max().date()}")

print("\n---Yearly spending---")
print(df.groupby("Year")["Amount"].sum().round(2).to_string())

print("\n--- Spending by Category ---")
print(df.groupby("Category")["Amount"]
      .agg(["sum","mean","count"])
      .sort_values("sum", ascending=False).round(2))

print("\n--- Festive Period Spending ---")
festive = df.groupby("FestivePeriod")["Amount"].sum().sort_values(ascending=False)
print(festive)

print("\n--- Payment Method Breakdown ---")
print(df.groupby("Use Chip")["Amount"]
        .agg(["count", "sum", "mean"]).round(2))

print("\n--- INSIGHT: Fraud rate by payment method ---")
for method in df["Use Chip"].unique():
    subset = df[df["Use Chip"] == method]
    fraud_count = subset[subset["Is Fraud?"] == "Yes"].shape[0]
    rate = fraud_count / len(subset) * 100
    print(f"  {method}: {fraud_count} fraud / {len(subset)} txns ({rate:.2f}%)")

print("\n--- Top 10 Cities by Spending ---")
print(df.groupby("Merchant City")["Amount"].sum()
       .sort_values(ascending=False).head(10))
    
print("\n--- Spending by Day of Week ---")
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print(df.groupby("DayOfWeek")["Amount"].sum()
      .reindex(dow_order).round(2))

print("\n--- Spending by Hour of Day ---")
print(df.groupby("Hour")["Amount"].sum()
      .sort_values(ascending=False).head(5))

top_cat = df.groupby("Category")["Amount"].sum().idxmax()
top_festive = df.groupby("FestivePeriod")["Amount"].sum().idxmax()
online = df[df["Use Chip"] == "Online Transaction"]
online_fraud_rate = online[online["Is Fraud?"] == "Yes"].shape[0] / len(online) * 100

insights = [
    f"Top spending category: {top_cat}",
    f"Highest spending festive period: {top_festive}",
    f"Online transaction fraud rate: {online_fraud_rate:.2f}%",
    f"Total fraud cases: {df[df['Is Fraud?'] == 'Yes'].shape[0]}",
]

with open("outputs/insights.txt", "w") as f:
    f.write("\n".join(insights))

print("\nInsights saved to outputs/insights.txt")    