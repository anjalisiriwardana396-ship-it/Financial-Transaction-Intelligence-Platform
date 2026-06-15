# Dataset Exploration

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

## Next Steps

* Convert Amount to numeric
* Create a proper Date column
* Handle missing values
* Create additional features such as DayOfWeek, Hour, and Season
* Perform EDA and fraud analysis
