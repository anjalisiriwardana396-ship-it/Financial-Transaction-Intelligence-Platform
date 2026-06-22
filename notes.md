# Phase 1 -Dataset Exploration

## Objective
Explore the raw credit card transaction dataset before cleaning or modeling.

## Dataset Overview
* Rows: 19,963
* Columns: 15
* Time span: 2002–2020
* Fraud cases: 27 (0.135%)

## Key Findings
### 1. Amount column needs cleaning

The `Amount` column is stored as a string and contains dollar signs.

Example:

$134.09
$38.48

This column must be converted to numeric before performing calculations or training models.

---

### 2. Missing values exist in important columns

| Column         | Missing Values |
| -------------- | -------------: |
| Merchant State |          1,317 |
| Zip            |          1,647 |
| Errors?        |         19,389 |

Most values in `Errors?` are missing, which probably means no error occurred. During cleaning I plan to replace missing values with `"No Error"`.

---

### 3. Fraud is extremely rare

Fraud breakdown:

* Legitimate: 19,936
* Fraud: 27
* Fraud rate: 0.135%

This indicates a severe class imbalance problem. Accuracy alone will be misleading for fraud detection. I will use SMOTE and metrics such as Recall, Precision, ROC-AUC, and Confusion Matrix.

---

### 4. Payment methods

Transaction counts:

* Swipe Transaction: 15,840
* Chip Transaction: 2,808
* Online Transaction: 1,315

Most transactions are swipe-based. I will later investigate whether fraud rates differ across payment methods.

# Phase 2 – Data Cleaning

## What I did
- Removed "$" from Amount column and converted it to float
- Created a proper Date column using Year, Month, Day
- Filled missing values in Merchant State, Zip, and Errors
- Removed duplicate transactions
- Created new features:
  - DayOfWeek
  - Hour
  - Festive Period
  - Merchant Category (from MCC)

## Why this was needed
Raw financial data is messy and not directly usable for analysis or ML.
Cleaning ensures:
- No missing value issues
- Correct data types for modeling
- Better feature engineering for insights and fraud detection

## Key output
- Dataset shape reduced/cleaned: (19963, XX columns)
- No duplicate rows found
- Ready for EDA and visualization phase

# Phase 3-EDA Insights
 
## Key Buisness Insights 
### Overall Spending
- Total transactions: 19,963
- Total spending: $1.62M
- Average transaction: $81.30
- Largest transaction: $1,409.40
- Data spans 2002–2020

---

### Yearly Trend
- Spending is relatively stable across years (~88k–99k range)
- No strong long-term growth or decline
- 2002 and 2020 have lower values (partial years likely)

---

### Category Insights
- Highest spending categories:
  - Other ($364K)
  - Pharmacy ($273K)
  - Grocery ($272K)
- Highest average transaction value:
  - Utilities (~$124)
  - Wholesale (~$116)

---

### Payment Method Behavior
- Swipe transactions dominate volume and total spending
- Online transactions have higher average value ($112 vs ~$80)

---

### Fraud Pattern Insight
- Online transactions have highest fraud rate (~1.14%)
- Chip transactions have lowest fraud rate (~0.04%)

---

### Geographic Insight
- Spending heavily concentrated in:
  - La Verne
  - Mira Loma
  - Monterey Park
- ONLINE transactions are a major category (important for fraud risk)

---

### Time-Based Behavior
- Spending is fairly evenly distributed across weekdays
- Hour 6 shows extremely high spending spike (~$875K) → needs validation (possible data issue or aggregation artifact)

---

### Seasonal / Festive Insight
- Strong festive spending spikes:
  - New Year: $148K
  - Christmas: $145K
  - Avurudu: $124K (important regional pattern)
- Seasonal behavior is clearly visible and meaningful for financial planning   