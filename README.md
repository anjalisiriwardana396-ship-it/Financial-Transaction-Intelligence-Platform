## Dataset Overview                                         -------------------

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
## Spending PREDICTION MODEL                                            -------------------------------
To forecast future credit card spending, a time-series regression model was built using historical transaction data (2002–2020).

### Approach
- Aggregated 19,963 transactions into monthly totals
- Engineered time-series features:
  - Lag1, Lag2, Lag3 (previous months' spending)
  - Rolling 3-month average
  - Month, Year, and festive indicators
- Trained a Linear Regression model

### Why this approach
Simple regression was chosen for interpretability and explainability in a financial context.

### Model Performance
- MAE (Mean Absolute Error): ~$706.80
- Average monthly spending: ~$8,000
- Error rate: ~8–9%

### Key Insight
- Model captures overall trend correctly
- Fails to capture sudden dips/spikes (expected limitation of linear models)
- Spending remains relatively stable in later years (~$7.6K–$7.8K forecasted)

### Business Value
- Helps estimate future cash flow
- Useful for personal finance tracking or banking analytics dashboards