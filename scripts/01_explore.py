import pandas as pd

df = pd.read_csv("data/User0_credit_card_transactions.csv")

print("=== FIRST 5 ROWS ===" )
print(df.head())

print("\n=== SHAPE ===")
print(df.shape)

print(("\n=== DATA TYPES ==="))
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== AMOUNT SAMPLE (check for $ signs) ===")
print(df["Amount"].head())

print("\n=== FRAUD BREAKDOWN ===")
print(df["Is Fraud?"].value_counts())
fraud_pct = df[df["Is Fraud?"] == "Yes"].shape[0] / len(df) * 100
print(f"Fraud %: {fraud_pct:.3f}%")

print("\n=== PAYMENT METHODS ===")
print(df["Use Chip"].value_counts())