# predict_demo.py
import os
import json
import joblib
import pandas as pd
from datetime import datetime

MODEL_DIR = "models"
PIPE_PATH = os.path.join(MODEL_DIR, "pipeline.joblib")
FEATURE_JSON = os.path.join(MODEL_DIR, "feature_list.json")

if not os.path.exists(PIPE_PATH):
    raise FileNotFoundError(" Model not found. Please run train_model.py first!")

# Load model & features
pipe = joblib.load(PIPE_PATH)
with open(FEATURE_JSON, "r") as f:
    feature_info = json.load(f)
features = feature_info.get("features", [])

# --- Example car data ---
example = {
    "yr_mfr": 2016,
    "kms_run": 42000,
    "fuel_type": "Petrol",
    "transmission": "Manual",
    "total_owners": 1,
    "make": "Maruti",
    "model": "Swift",
    "city": "Bengaluru",
    "body_type": "Hatchback",
    "car_rating": "Good",
    "warranty_avail": 0,
    "original_price": 600000
}

# Compute engineered fields
now_year = datetime.now().year
example["age"] = now_year - example["yr_mfr"]
example["kms_per_year"] = example["kms_run"] / example["age"]

# Ensure all expected features exist
for f in features:
    if f not in example:
        example[f] = None

X = pd.DataFrame([{f: example.get(f) for f in features}])
print("\n Input Data:")
print(X)

# Predict
pred = pipe.predict(X)[0]
print(f"\nPredicted Used Car Price: Rs. {round(pred, 2)}")
