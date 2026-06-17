import pandas as pd

df = pd.read_csv("data/User0_credit_card_transactions.csv")
print("Shape Before:", df.shape)

# FIX 1: Remove "$" from Amount and convert to a real number
df["Amount"] = df["Amount"].str.replace("$", "",regex=False).astype(float)
print("Amount fixed:", df["Amount"].head().tolist())

# FIX 2: Build one Date column from 3 separate columns (Year, Month, Day)
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

# FIX 3: Fill missing Merchant State
df["Merchant State"] = df["Merchant State"].fillna("Unknown")

# FIX 4: Fill missing Zip
df["Zip"] = df["Zip"].fillna(0).astype(int)

# FIX 5: Fill missing Errors
df["Errors?"] = df["Errors?"].fillna("No Error")

# FIX 6: Remove duplicate rows
print("Duplicates found:", df.duplicated().sum())
df = df.drop_duplicates()

# ADD EXTRA COLUMN 1: Day name
df["DayOfWeek"] = df["Date"].dt.day_name()

# ADD EXTRA COLUMN 2: Hour of transaction
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour

# ADD EXTRA COLUMN 3: Festive period (mapping relevant to Sri Lanka)
def get_festive_period(month):
    if month == 4:              return "Avurudu"
    elif month in [12]:         return "Christmas"
    elif month in [1]:          return "New Year"
    elif month == 2:            return "Valentine"
    else:                       return "Normal"

df["FestivePeriod"] = df["Month"].apply(get_festive_period)

# ADD EXTRA COLUMN 4: Readable category from MCC code
mcc_map = {
    5411: "Grocery",     5912: "Pharmacy",
    5300: "Wholesale",   5541: "Gas Station",
    5311: "Dept Store",  5812: "Restaurant",
    5499: "Misc Food",   5942: "Book Store",
    7538: "Auto Repair", 4900: "Utilities",
    5661: "Shoe Store",  5651: "Clothing",
    5251: "Hardware",    5200: "Home Supply",
    5999: "Misc Retail"
}
df["Category"] = df["MCC"].map(mcc_map).fillna("Other")

print("\n=== FINAL CHECK ===")
print("Shape AFTER:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Save the cleaned file so every future phase uses this clean version
df.to_csv("outputs/cleaned_transactions.csv", index=False)
print("\nSaved to outputs/cleaned_transactions.csv")