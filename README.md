## Dataset Overview

* 19,963 credit card transactions
* 15 original features
* Transaction history from 2002–2020
* Includes merchant details, payment methods, MCC codes, and fraud labels

## Initial Findings

* The `Amount` column contains currency symbols and requires preprocessing.
* Missing values exist in `Merchant State`, `Zip`, and `Errors?`.
* Fraud transactions are extremely rare: only **27 out of 19,963 transactions (0.135%)**.
* Swipe transactions account for the majority of all transactions.

## Challenges

The fraud detection task presents a severe class imbalance problem. Because fraudulent transactions represent only 0.135% of the data, accuracy alone is not a reliable metric. The project addresses this by applying SMOTE on the training data and evaluating the model using precision, recall, ROC-AUC, and confusion matrices.
## Data Cleaning Performed

The following preprocessing steps were applied to make the dataset analysis-ready:

- Removed `$` symbol from `Amount` and converted it to numeric type
- Combined `Year`, `Month`, and `Day` into a single `Date` column
- Filled missing values:
  - `Merchant State` → "Unknown"
  - `Zip` → 0
  - `Errors?` → "No Error"
- Removed duplicate transactions
- Created new features:
  - `DayOfWeek` (weekday pattern analysis)
  - `Hour` (transaction time behavior)
  - `Festive period` (festive spending patterns)
  - `Category` (mapped from MCC codes)