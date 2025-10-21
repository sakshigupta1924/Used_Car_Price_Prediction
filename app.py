# app.py (replace your existing app.py with this)
from flask import Flask, render_template, request, jsonify
import joblib, os, json, math
import pandas as pd
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "pipeline.joblib")
FEATURE_PATH = os.path.join("models", "feature_list.json")
DATA_SAMPLE = r"C:\Users\gunus\OneDrive\Desktop\used car price pred\New folder\Used_Car_Price_Prediction.csv"  # sample to extract dropdowns

# load model & features
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Pipeline not found at {MODEL_PATH}. Run training first.")
model = joblib.load(MODEL_PATH)
with open(FEATURE_PATH, "r") as f:
    expected_features = json.load(f).get("features", [])

# prepare dropdown options from sample CSV (safe)
dropdowns = {}
if os.path.exists(DATA_SAMPLE):
    try:
        df_sample = pd.read_csv(DATA_SAMPLE, nrows=5000)
        dropdowns["make"] = sorted(df_sample["make"].dropna().unique().tolist()) if "make" in df_sample.columns else []
        dropdowns["model"] = sorted(df_sample["model"].dropna().unique().tolist()) if "model" in df_sample.columns else []
        dropdowns["city"] = sorted(df_sample["city"].dropna().unique().tolist()) if "city" in df_sample.columns else []
        dropdowns["fuel_type"] = sorted(df_sample["fuel_type"].dropna().unique().tolist()) if "fuel_type" in df_sample.columns else ["Petrol","Diesel","CNG","Electric"]
        dropdowns["transmission"] = sorted(df_sample["transmission"].dropna().unique().tolist()) if "transmission" in df_sample.columns else ["Manual","Automatic"]
        dropdowns["body_type"] = sorted(df_sample["body_type"].dropna().unique().tolist()) if "body_type" in df_sample.columns else []
    except Exception as e:
        dropdowns = {}
else:
    dropdowns = {}

def safe_float(x):
    try:
        if x is None or str(x).strip()=="":
            return np.nan
        return float(x)
    except:
        return np.nan

def feature_engineer(df):
    current_year = pd.Timestamp.now().year
    if "yr_mfr" in df.columns:
        df["yr_mfr"] = pd.to_numeric(df["yr_mfr"], errors="coerce")
        df["age"] = current_year - df["yr_mfr"]
        df["age"] = df["age"].where((df["age"] >= 0) & (df["age"] < 100), np.nan)
    else:
        df["age"] = np.nan
    if "kms_run" in df.columns:
        df["kms_run"] = pd.to_numeric(df["kms_run"], errors="coerce")
        df.loc[df["kms_run"] < 0, "kms_run"] = np.nan
    else:
        df["kms_run"] = np.nan
    # kms_per_year
    def kpy(r):
        kms = r.get("kms_run", np.nan)
        age = r.get("age", np.nan)
        try:
            if np.isnan(kms) or np.isnan(age) or age <= 0:
                return kms if not (math.isnan(kms)) else np.nan
            return kms/age
        except:
            return np.nan
    df["kms_per_year"] = df.apply(kpy, axis=1)
    # warranty normalize
    if "warranty_avail" in df.columns:
        df["warranty_avail"] = df["warranty_avail"].astype(str).str.lower().map({"true":"1","false":"0","yes":"1","no":"0","1":"1","0":"0"}).fillna("0")
        df["warranty_avail"] = pd.to_numeric(df["warranty_avail"], errors="coerce").fillna(0).astype(int)
    else:
        df["warranty_avail"] = 0
    return df

@app.route("/")
def home():
    return render_template("index.html")

# API returning dropdown options for the frontend
@app.route("/api/options")
def api_options():
    return jsonify(dropdowns)

# JSON API for predictions (useful for JS fetch)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    # build input dict
    input_dict = {}
    for feat in expected_features:
        input_dict[feat] = data.get(feat, None)
    # Also accept common keys if expected_features don't include all
    for k in ["yr_mfr","kms_run","fuel_type","transmission","total_owners","make","model","city","body_type","car_rating","warranty_avail","original_price"]:
        if k not in input_dict:
            input_dict[k] = data.get(k, None)
    df_in = pd.DataFrame([input_dict])
    df_in = feature_engineer(df_in)

    # ensure expected features present
    for feat in expected_features:
        if feat not in df_in.columns:
            df_in[feat] = np.nan
    df_in = df_in[expected_features] if expected_features else df_in
    try:
        pred = model.predict(df_in)[0]
        return jsonify({"predicted_price": float(round(float(pred),2))})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# old form endpoint (keeps compatibility)
@app.route("/predict", methods=["POST"])
def predict_form():
    try:
        form = request.form.to_dict()
        # convert types carefully
        example = {
            "yr_mfr": safe_float(form.get("yr_mfr")),
            "kms_run": safe_float(form.get("kms_run")),
            "fuel_type": form.get("fuel_type"),
            "transmission": form.get("transmission"),
            "total_owners": safe_float(form.get("total_owners")),
            "make": form.get("make"),
            "model": form.get("model"),
            "city": form.get("city"),
            "body_type": form.get("body_type"),
            "car_rating": form.get("car_rating"),
            "warranty_avail": form.get("warranty_avail"),
            "original_price": safe_float(form.get("original_price"))
        }
        df_in = pd.DataFrame([example])
        df_in = feature_engineer(df_in)
        for feat in expected_features:
            if feat not in df_in.columns:
                df_in[feat] = np.nan
        df_in = df_in[expected_features] if expected_features else df_in
        pred = model.predict(df_in)[0]
        return render_template("index.html", prediction_text=f"Predicted Used Car Price: Rs. {round(pred,2)}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
