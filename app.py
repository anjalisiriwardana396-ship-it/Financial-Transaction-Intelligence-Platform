from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ── Load all saved models when app starts ────────────────────
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

# Load behavioral profile if it exists
try:
    with open("model/behavioral_profile.pkl", "rb") as f:
        behavioral_profile = pickle.load(f)
except FileNotFoundError:
    behavioral_profile = {}    

df = pd.read_csv("outputs/cleaned_transactions.csv")
print("All models loaded successfully")

# ── ROUTE 1: Home page ───────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

# ── ROUTE 2: KPI Summary ─────────────────────────────────────
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

# ── ROUTE 3: Business Insights ───────────────────────────────
@app.route("/insights")
def insights():
    online = df[df["Use Chip"] == "Online Transaction"]
    online_fraud_rate = (online[online["Is Fraud?"] == "Yes"].shape[0]
                         / len(online) * 100)
    top_cat     = df.groupby("Category")["Amount"].sum().idxmax()
    top_festive = df.groupby("FestivePeriod")["Amount"].sum().idxmax()
    yearly      = df.groupby("Year")["Amount"].sum()
    peak_year   = int(yearly.idxmax())
    peak_amount = round(float(yearly.max()), 2)
    return jsonify({
        "online_fraud_rate":  round(online_fraud_rate, 2),
        "top_category":       top_cat,
        "top_festive":        top_festive,
        "peak_year":          peak_year,
        "peak_year_amount":   peak_amount
    })

# ── ROUTE 4: Chart Data ──────────────────────────────────────
@app.route("/chart_data")
def chart_data():
    yearly = df.groupby("Year")["Amount"].sum().round(2)
    cats   = (df.groupby("Category")["Amount"].sum()
                .sort_values(ascending=False).round(2))
    fraud_by_method = {}
    for method in df["Use Chip"].unique():
        subset = df[df["Use Chip"] == method]
        fraud_by_method[method] = int(
            subset[subset["Is Fraud?"] == "Yes"].shape[0])

    # Use monthly averages for fair comparison
    # Normal covers ~8 months, festive periods cover 1 month each
    festive_raw = df.groupby("FestivePeriod")["Amount"].sum()
    months_map  = {"Normal":8,"Avurudu":1,"Christmas":1,"New Year":1,"Valentine":1}
    festive_avg = {p: round(float(amt)/months_map.get(p,1), 2)
                   for p, amt in festive_raw.items()}

    return jsonify({
        "yearly_labels":   list(yearly.index.astype(str)),
        "yearly_values":   list(yearly.values),
        "category_labels": list(cats.index),
        "category_values": list(cats.values),
        "fraud_methods":   fraud_by_method,
        "festive_labels":  list(festive_avg.keys()),
        "festive_values":  list(festive_avg.values())
    })

# ── ROUTE 5: Spending Prediction ─────────────────────────────
@app.route("/predict_spending", methods=["POST"])
def predict_spending():
    data         = request.get_json()
    months_ahead = int(data.get("months_ahead", 1))
    if months_ahead < 1 or months_ahead > 12:
        return jsonify({"error": "Please enter between 1 and 12 months"}), 400

    recent     = list(month_info["recent_amounts"])
    last_month = month_info["last_month_num"]
    last_year  = month_info["last_year"]
    last_idx   = month_info["last_idx"]

    def get_festive_num(m):
        if m == 4:    return 4
        elif m == 12: return 3
        elif m == 1:  return 2
        elif m == 2:  return 1
        else:         return 0

    preds = []
    for i in range(months_ahead):
        lag1     = recent[-1]
        lag2     = recent[-2]
        lag3     = recent[-3]
        rolling3 = np.mean(recent[-3:])
        m        = ((last_month + i) % 12) + 1
        y        = last_year + ((last_month + i) // 12)
        festive  = get_festive_num(int(m))
        idx      = last_idx + i + 1
        row      = pd.DataFrame(
                     [[lag1, lag2, lag3, rolling3, m, y, festive, idx]],
                     columns=month_info["features"])
        pred     = spending_model.predict(row)[0]
        preds.append(round(float(pred), 2))
        recent.append(pred)

    return jsonify({
        "months_ahead":   months_ahead,
        "predictions":    preds,
        "predicted_next": preds[0]
    })

# ── ROUTE 6: Fraud Prediction ────────────────────────────────
@app.route("/predict_fraud", methods=["POST"])
def predict_fraud():
    data      = request.get_json()
    text_cols = ["Use Chip", "Merchant State", "Errors?", "DayOfWeek"]
    for col in text_cols:
        val = data.get(col, "Unknown")
        try:
            data[col] = int(fraud_encoders[col].transform([val])[0])
        except Exception:
            data[col] = 0
    row  = pd.DataFrame(
             [[float(data.get(f, 0)) for f in fraud_features]],
             columns=fraud_features)
    pred = fraud_model.predict(row)[0]
    prob = fraud_model.predict_proba(row)[0][1]
    return jsonify({
        "is_fraud":          bool(pred),
        "fraud_probability": round(float(prob) * 100, 1),
        "verdict":           "FRAUD DETECTED" if pred else "Looks Legitimate"
    })

# ── ROUTE 7: Comparison Data ─────────────────────────────────
@app.route("/comparison_data")
def comparison_data():
    total       = len(df)
    total_spent = df["Amount"].sum()

    method_stats = []
    for method in ["Online Transaction", "Chip Transaction", "Swipe Transaction"]:
        subset      = df[df["Use Chip"] == method]
        fraud_count = subset[subset["Is Fraud?"] == "Yes"].shape[0]
        rate        = round(fraud_count / len(subset) * 100, 3)
        method_stats.append({
            "method":      method,
            "total":       len(subset),
            "fraud_count": fraud_count,
            "fraud_rate":  rate
        })

    cat_totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    cat_stats  = []
    for cat, amt in cat_totals.head(5).items():
        cat_stats.append({
            "category":   cat,
            "amount":     round(float(amt), 2),
            "percentage": round(float(amt) / total_spent * 100, 1)
        })

    yearly      = df.groupby("Year")["Amount"].sum()
    avg_yearly  = round(float(yearly.mean()), 2)
    peak_year   = int(yearly.idxmax())
    peak_amt    = round(float(yearly.max()), 2)
    peak_vs_avg = round((peak_amt - avg_yearly) / avg_yearly * 100, 1)

    festive       = df.groupby("FestivePeriod")["Amount"].sum()
    normal_avg    = round(float(festive["Normal"]) / 8, 2)
    festive_stats = {}
    for period, amt in festive.items():
        months = 8 if period == "Normal" else 1
        festive_stats[period] = {
            "total":       round(float(amt), 2),
            "monthly_avg": round(float(amt) / months, 2)
        }

    model_comparison = [
        {"model": "Random Forest",       "roc_auc": 0.9809, "recall": 0.200},
        {"model": "XGBoost",             "roc_auc": 0.9342, "recall": 0.200},
        {"model": "Logistic Regression", "roc_auc": 0.8410, "recall": 0.400}
    ]

    return jsonify({
        "method_stats":     method_stats,
        "cat_stats":        cat_stats,
        "avg_yearly_spend": avg_yearly,
        "peak_year":        peak_year,
        "peak_amt":         peak_amt,
        "peak_vs_avg":      peak_vs_avg,
        "festive_stats":    festive_stats,
        "normal_monthly":   normal_avg,
        "model_comparison": model_comparison
    })

# ── Analytics data ───────────────────────────────────────────
@app.route("/analytics_data")
def analytics_data():
    cols = ["Date","Amount","Use Chip","Merchant City",
            "Category","Hour","DayOfWeek","FestivePeriod","Is Fraud?"]
    rows = df[cols].head(500).to_dict(orient="records")
    return jsonify({"rows": rows})

# ── Export routes ────────────────────────────────────────────
@app.route("/export_csv")
def export_csv():
    from flask import send_file
    return send_file("outputs/cleaned_transactions.csv",
                     as_attachment=True,
                     download_name="cleaned_transactions.csv")

@app.route("/export_outliers")
def export_outliers():
    from flask import send_file
    return send_file("outputs/outliers.csv",
                     as_attachment=True,
                     download_name="outliers.csv")

@app.route("/export_models")
def export_models():
    from flask import send_file
    return send_file("outputs/model_comparison.csv",
                     as_attachment=True,
                     download_name="model_comparison.csv")

# ── Serve chart images ───────────────────────────────────────
@app.route("/chart_image/<filename>")
def chart_image(filename):
    from flask import send_from_directory
    return send_from_directory("outputs", filename)

# ── Load behavioral profile ──────────────────────────────────
with open("model/behavioral_profile.pkl", "rb") as f:
    behavioral_profile = pickle.load(f)

# ── ROUTE: Behavioral profile ────────────────────────────────
@app.route("/behavioral_profile")
def get_behavioral_profile():
    return jsonify(behavioral_profile)

# ── ROUTE: Enhanced fraud check with behavioral context ──────
@app.route("/predict_fraud_behavioral", methods=["POST"])
def predict_fraud_behavioral():
    data = request.get_json()

    # Encode text columns
    text_cols = ["Use Chip", "Merchant State", "Errors?", "DayOfWeek"]
    for col in text_cols:
        val = data.get(col, "Unknown")
        try:
            data[col] = int(fraud_encoders[col].transform([val])[0])
        except Exception:
            data[col] = 0

    amount = float(data.get("Amount", 0))
    hour   = int(data.get("Hour", 0))
    method = data.get("original_method", "Swipe Transaction")

    # Run the model
    row  = pd.DataFrame([[float(data.get(f, 0)) for f in fraud_features]],
                         columns=fraud_features)
    pred = fraud_model.predict(row)[0]
    prob = fraud_model.predict_proba(row)[0][1]

    # ── Behavioral context flags ─────────────────────────────
    flags = []

    # Flag 1: Amount deviation
    avg   = behavioral_profile["avg_spend"]
    std   = behavioral_profile["std_spend"]
    high  = behavioral_profile["high_spend_threshold"]
    extreme = behavioral_profile["extreme_spend_threshold"]
    if amount > extreme:
        flags.append(f"Amount ${amount:.0f} is extreme (>{extreme:.0f} = 3σ above your baseline)")
    elif amount > high:
        flags.append(f"Amount ${amount:.0f} is above normal range (>{high:.0f} = 2σ above baseline)")

    # Flag 2: Unusual hour
    unusual_pct = behavioral_profile["unusual_hours_pct"]
    if hour < 9 or hour > 18:
        flags.append(f"Hour {hour}:00 is outside normal business hours ({unusual_pct}% of transactions are at unusual times)")

    # Flag 3: Online risk
    online_ratio = behavioral_profile["online_ratio"]
    if method == "Online Transaction":
        flags.append(f"Online transaction — historically {online_ratio}% of your transactions are online, and online fraud rate is 1.14%")

    # Flag 4: Amount vs fraud baseline
    fraud_avg = behavioral_profile["fraud_avg_amount"]
    legit_avg = behavioral_profile["legit_avg_amount"]
    if abs(amount - fraud_avg) < abs(amount - legit_avg):
        flags.append(f"Amount closer to historical fraud average (${fraud_avg}) than legitimate average (${legit_avg:.0f})")

    # Deviation score
    deviation = round(abs(amount - avg) / std, 2) if std > 0 else 0

    return jsonify({
        "is_fraud":          bool(pred),
        "fraud_probability": round(float(prob) * 100, 1),
        "verdict":           "FRAUD DETECTED" if pred else "Looks Legitimate",
        "behavioral_flags":  flags,
        "deviation_score":   deviation,
        "amount_context": {
            "your_avg":   avg,
            "this_amount": amount,
            "deviation":  deviation
        }
    })

# ── This must always be the very last line ───────────────────
if __name__ == "__main__":
    app.run(debug=True)