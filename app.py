from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
# Initialize the Flask application instance
# __name__ helps Flask locate templates and static resources correctly

# ─────────────────────────────────────────────────────────────
# Load all pre-trained models and supporting artifacts at startup
# Reason: Loading models once improves performance and avoids
# repeated disk I/O on every request
# ─────────────────────────────────────────────────────────────
with open("model/spending_model.pkl", "rb") as f:
    spending_model = pickle.load(f)

with open("model/month_info.pkl", "rb") as f:
    month_info = pickle.load(f)

with open("model/fraud_model.pkl", "rb") as f:
    fraud_model = pickle.load(f)

with open("model/fraud_encoders.pkl", "rb") as f:
    fraud_encoders = pickle.load(f)

with open("model/fraud_features.pkl", "rb") as f:
    fraud_features = pickle.load(f)

with open("model/fraud_best_name.pkl", "rb") as f:
    best_model_name = pickle.load(f)

# Load cleaned transaction dataset for analytics and KPI generation
df = pd.read_csv("outputs/cleaned_transactions.csv")

print("All models loaded successfully")

# ─────────────────────────────────────────────────────────────
# ROUTE 1: Home Page
# Serves the main dashboard UI
# ─────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")
# Renders the HTML template from the templates directory

# ─────────────────────────────────────────────────────────────
# ROUTE 2: KPI Summary Endpoint
# Returns high-level business metrics for dashboard display
# ─────────────────────────────────────────────────────────────
@app.route("/summary")
def summary():
    fraud_count = int(df[df["Is Fraud?"] == "Yes"].shape[0])
    total = len(df)

    return jsonify({
        "total_transactions": total,
        "total_spent":        round(df["Amount"].sum(), 2),
        "avg_transaction":    round(df["Amount"].mean(), 2),
        "fraud_count":        fraud_count,
        "fraud_rate":         round(fraud_count / total * 100, 3),
        "best_fraud_model":   best_model_name
    })
# jsonify converts Python dictionaries into JSON responses for the frontend

# ─────────────────────────────────────────────────────────────
# ROUTE 3: Business Insights Endpoint
# Provides aggregated patterns and trends from transaction data
# ─────────────────────────────────────────────────────────────
@app.route("/insights")
def insights():
    online = df[df["Use Chip"] == "Online Transaction"]

    # Fraud rate for online transactions
    online_fraud_rate = (
        online[online["Is Fraud?"] == "Yes"].shape[0]
        / len(online) * 100
    )

    # Identify top spending categories and festive periods
    top_cat = df.groupby("Category")["Amount"].sum().idxmax()
    top_festive = df.groupby("FestivePeriod")["Amount"].sum().idxmax()

    # Determine highest spending year
    yearly = df.groupby("Year")["Amount"].sum()
    peak_year = int(yearly.idxmax())
    peak_amount = round(float(yearly.max()), 2)

    return jsonify({
        "online_fraud_rate":  round(online_fraud_rate, 2),
        "top_category":       top_cat,
        "top_festive":        top_festive,
        "peak_year":          peak_year,
        "peak_year_amount":   peak_amount
    })

# ─────────────────────────────────────────────────────────────
# ROUTE 4: Chart Data Endpoint
# Supplies structured data for frontend visualizations
# ─────────────────────────────────────────────────────────────
@app.route("/chart_data")
def chart_data():

    # Year-wise total spending (line chart)
    yearly = df.groupby("Year")["Amount"].sum().round(2)

    # Category-wise total spending (bar chart)
    cats = (
        df.groupby("Category")["Amount"].sum()
        .sort_values(ascending=False)
        .round(2)
    )

    # Fraud count grouped by payment method
    fraud_by_method = {}
    for method in df["Use Chip"].unique():
        subset = df[df["Use Chip"] == method]
        fraud_by_method[method] = int(
            subset[subset["Is Fraud?"] == "Yes"].shape[0]
        )

    # Spending distribution across festive periods
    festive = df.groupby("FestivePeriod")["Amount"].sum().round(2)

    return jsonify({
        "yearly_labels":    list(yearly.index.astype(str)),
        "yearly_values":    list(yearly.values),
        "category_labels":  list(cats.index),
        "category_values":  list(cats.values),
        "fraud_methods":    fraud_by_method,
        "festive_labels":   list(festive.index),
        "festive_values":   list(festive.values)
    })

# ─────────────────────────────────────────────────────────────
# ROUTE 5: Spending Forecast Endpoint
# Predicts future spending based on historical patterns
# ─────────────────────────────────────────────────────────────
@app.route("/predict_spending", methods=["POST"])
def predict_spending():

    # Parse JSON payload from client request
    data = request.get_json()

    months_ahead = int(data.get("months_ahead", 1))

    # Validate input range for prediction horizon
    if months_ahead < 1 or months_ahead > 12:
        return jsonify({"error": "Please enter between 1 and 12 months"}), 400
    # HTTP 400 indicates a client-side invalid request

    # Extract historical monthly spending context
    recent     = list(month_info["recent_amounts"])
    last_month = month_info["last_month_num"]
    last_year  = month_info["last_year"]
    last_idx   = month_info["last_idx"]

    # Encode seasonal/festive impact as numerical feature
    def get_festive_num(m):
        if m == 4:    return 4
        elif m == 12: return 3
        elif m == 1:  return 2
        elif m == 2:  return 1
        else:         return 0

    preds = []

    # Iteratively predict future monthly spending
    for i in range(months_ahead):

        lag1     = recent[-1]
        lag2     = recent[-2]
        lag3     = recent[-3]
        rolling3 = np.mean(recent[-3:])

        # Compute future month and year
        m = ((last_month + i) % 12) + 1
        y = last_year + ((last_month + i) // 12)

        festive = get_festive_num(int(m))
        idx     = last_idx + i + 1

        # Construct feature vector for model input
        row = pd.DataFrame(
            [[lag1, lag2, lag3, rolling3, m, y, festive, idx]],
            columns=month_info["features"]
        )

        # Generate prediction
        pred = spending_model.predict(row)[0]
        preds.append(round(float(pred), 2))

        # Append prediction for recursive forecasting
        recent.append(pred)

    return jsonify({
        "months_ahead":   months_ahead,
        "predictions":    preds,
        "predicted_next": preds[0]
    })

# ─────────────────────────────────────────────────────────────
# ROUTE 6: Fraud Detection Endpoint
# Predicts whether a transaction is fraudulent
# ─────────────────────────────────────────────────────────────
@app.route("/predict_fraud", methods=["POST"])
def predict_fraud():

    data = request.get_json()

    # Encode categorical variables using pre-fitted encoders
    text_cols = ["Use Chip", "Merchant State", "Errors?", "DayOfWeek"]

    for col in text_cols:
        val = data.get(col, "Unknown")
        try:
            data[col] = int(fraud_encoders[col].transform([val])[0])
        except Exception:
            # Handle unseen categories gracefully
            data[col] = 0

    # Construct feature vector in required model format
    row = pd.DataFrame(
        [[float(data.get(f, 0)) for f in fraud_features]],
        columns=fraud_features
    )

    # Generate fraud prediction and probability score
    pred = fraud_model.predict(row)[0]
    prob = fraud_model.predict_proba(row)[0][1]

    return jsonify({
        "is_fraud":          bool(pred),
        "fraud_probability": round(float(prob) * 100, 1),
        "verdict":           "FRAUD DETECTED" if pred else "Looks Legitimate"
    })

# ─────────────────────────────────────────────────────────────
# Application entry point
# Runs the Flask development server
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
# debug=True enables auto-reload and detailed error messages
# Should only be used in development, not production environments