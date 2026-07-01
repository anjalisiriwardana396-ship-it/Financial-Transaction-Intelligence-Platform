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
- Seasonal behavior is clearly visible and Meaningful for financial behaviour.                        
# Phase 6 – Spending Prediction Model (Learning Notes)
### What I learned
- How to convert transaction-level data into time-series data
- How lag features work (Lag1, Lag2, Lag3)
- Why rolling averages help smooth volatility
- Why time-series split is not random split

### Key Concepts
- Lag features = previous values used for prediction
- Rolling mean = trend smoothing
- Time-based split = prevents data leakage

### Model Behavior
- Linear regression captures trend but not sudden changes
- MAE ~ $706 is acceptable for financial forecasting
- Predictions stay stable due to linear nature of model

### Limitations observed
- Cannot detect sudden spending spikes
- Sensitive to long-term linear assumption                                         ###
# Phase 7 - Fraud Detection Model
## Objective

Build a machine learning model capable of predicting whether a credit card transaction is fraudulent.

---

## Problem

The dataset contains:

- Total transactions: 19,963
- Fraud transactions: 27
- Fraud percentage: 0.135%

This creates a severe class imbalance problem.

A model predicting "No Fraud" for every transaction would achieve approximately **99.8% accuracy**, while detecting zero fraudulent transactions.

Therefore, accuracy is not an appropriate evaluation metric.

---

## Solution

The workflow followed was:

1. Select useful transaction features.
2. Encode categorical variables using LabelEncoder.
3. Split the dataset into training and testing sets using stratified sampling.
4. Apply SMOTE only to the training data.
5. Train multiple classification models.
6. Compare model performance.
7. Save the best-performing model.

---

## Why SMOTE?

SMOTE (Synthetic Minority Oversampling Technique) creates synthetic fraud samples to balance the training dataset.

Correct workflow:

Original Dataset

↓

Train/Test Split

↓

SMOTE on Training Data Only

↓

Train Model

↓

Evaluate on Original Test Data

Applying SMOTE before splitting would introduce data leakage and produce misleading evaluation results.

---

## Models Compared

### Logistic Regression

Advantages:
- Simple
- Fast
- Easy to interpret

Result:

- Precision: 0.003
- Recall: 0.400
- ROC-AUC: 0.8410

---

### Random Forest

Advantages:
- Handles nonlinear relationships
- Robust against overfitting
- Provides feature importance

Result:

- Precision: 0.250
- Recall: 0.200
- F1-score: 0.222
- ROC-AUC: 0.9809

Selected as the final model.

---

### XGBoost

Advantages:
- Powerful boosting algorithm
- Often achieves excellent performance

Result:

- Precision: 0.200
- Recall: 0.200
- ROC-AUC: 0.9342

---

## Evaluation Metrics Learned

### Precision

Among all transactions predicted as fraud,
how many were actually fraud?

Higher precision means fewer false alarms.

---

### Recall

Among all actual fraud transactions,
how many were detected?

Higher recall means fewer fraud cases are missed.

---

### F1-score

Balances Precision and Recall.

Useful when dealing with imbalanced datasets.

---

### ROC-AUC

Measures the model's ability to distinguish fraudulent and legitimate transactions.

- 1.0 = Perfect
- 0.5 = Random guessing

Random Forest achieved:

ROC-AUC = **0.9809**

---

## Feature Importance

Most influential features:

1. Hour
2. MCC
3. Merchant State
4. Payment Method
5. Month

Least influential feature:

- Errors?

---

## Lessons Learned

- Never rely on accuracy when classes are highly imbalanced.
- Always split the dataset before applying SMOTE.
- Compare multiple models instead of selecting one immediately.
- ROC-AUC provides a better evaluation than accuracy for fraud detection.
- Feature importance helps explain why the model makes predictions.

---

## Final Outcome

Successfully built a fraud detection pipeline that:

- Handles severe class imbalance correctly.
- Prevents data leakage.
- Compares multiple machine learning models.
- Selects the best model objectively.
- Saves the trained model for deployment in the Flask application.                                                                                                                                       
---

# Phase 8 — Flask App Notes

## Purpose of this phase

This phase connects machine learning models to a web application using Flask.

It acts as the **deployment layer** of the project.

---

## Architecture Flow

Frontend (HTML Dashboard)
        ↓
Flask Backend (app.py)
        ↓
Machine Learning Models (pickle files)
        ↓
JSON Response → Frontend Updates UI

---

## Routes Summary

### 1. Home Route (`/`)
- Loads dashboard UI (index.html)

### 2. Summary API (`/summary`)
Returns:
- Total transactions
- Total spending
- Fraud count
- Fraud rate
- Best model used

---

### 3. Insights API (`/insights`)
Returns:
- Online fraud rate
- Top spending category
- Peak festive period
- Peak spending year

---

### 4. Chart Data API (`/chart_data`)
Used for visualizations:
- Yearly spending trend
- Category distribution
- Fraud by payment method
- Festive spending breakdown

---

### 5. Spending Prediction API (`/predict_spending`)

Input:
- months_ahead (1–12)

Process:
- Uses trained Linear Regression model
- Builds lag features:
  - Lag1, Lag2, Lag3
  - Rolling mean
- Predicts future monthly spending iteratively

Output:
- Future spending values

---

### 6. Fraud Prediction API (`/predict_fraud`)

Input:
- Transaction features (amount, MCC, merchant, time, etc.)

Process:
- Encodes categorical values using saved LabelEncoders
- Uses trained classification model
- Outputs probability + final decision

Output:
- Fraud / Not Fraud
- Fraud probability (%)
- Verdict label

---

## Important Technical Concepts

### 1. Model Persistence
- Models saved using pickle
- Loaded once when Flask starts

### 2. Label Encoding Consistency
- Same encoders used in training and inference
- Prevents mismatch in categorical data

### 3. Time-Series Prediction Logic
- Uses lag-based features
- Prevents future leakage
- Forecasts step-by-step

### 4. Class Imbalance Handling (Fraud Model)
- SMOTE used during training only
- Prevents synthetic data leakage into test set

---

## Key Learning Outcome

This phase demonstrates:

- How ML models are deployed in real applications
- How APIs connect backend ML to frontend UI
- How real-time predictions are served using Flask

---

## Final System Status

✔ Fraud Detection API working  
✔ Spending Prediction API working  
✔ Dashboard APIs integrated  
✔ Models loaded at runtime  
✔ JSON communication established     