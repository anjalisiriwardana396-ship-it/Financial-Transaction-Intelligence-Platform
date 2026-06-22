## Dataset Overview                       -------------------

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
## Data Cleaning Performed                          ---------------------------
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
## Key EDA Insights -------------------                                                         
- Total spending across dataset :1.62M across 19963 transactions                                        - Average transaction value: $81.30
- Spending is relatively stable across years (no major long-term growth trend)

### Category Behavior
- Highest spending occurs in:
  - Other, Pharmacy, Grocery categories
- Utilities and Wholesale have higher average transaction values

### Fraud Behavior
- Online transactions show the highest fraud rate (~1.14%)
- Chip transactions are the most secure (~0.04% fraud rate)

### Time-Based Insights
- Spending is distributed fairly evenly across weekdays
- Significant spending spike observed at Hour 6 (requires further investigation)

### Seasonal Trends
- Strong spending peaks during:
  - New Year
  - Christmas
  - Avurudu (regional seasonal effect)

### Geographic Insights
- Spending is concentrated in a few cities like La Verne, Mira Loma, and Monterey Park
- Online transactions also form a significant portion of activity