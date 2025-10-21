# train_model.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn

# ---------- CONFIG ----------
DATA_PATH = r"C:\Users\gunus\OneDrive\Desktop\used car price pred\New folder\Used_Car_Price_Prediction.csv"
MODEL_DIR = "models"
PIPE_PATH = os.path.join(MODEL_DIR, "pipeline.joblib")
FEATURE_JSON = os.path.join(MODEL_DIR, "feature_list.json")
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
print("scikit-learn version:", sklearn.__version__)
print("Dataset path:", DATA_PATH)

# ---------- FEATURES ----------
features = [
    "yr_mfr",
    "kms_run",
    "fuel_type",
    "transmission",
    "total_owners",
    "make",
    "model",
    "city",
    "body_type",
    "car_rating",
    "warranty_avail",
    "original_price"
]
target = "sale_price"

# ---------- HANDLE OneHotEncoder VERSION ----------
ohe_kwargs = {}
try:
    OneHotEncoder(sparse_output=False)
    ohe_kwargs["sparse_output"] = False
except TypeError:
    ohe_kwargs["sparse"] = False

# ---------- LOAD DATA ----------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f" CSV not found at: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("Data loaded successfully!", df.shape)

# ---------- CLEAN DATA ----------
df = df.drop_duplicates().reset_index(drop=True)
for col in ["kms_run", "sale_price", "original_price"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.replace(r"[^\d\.]", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- FEATURE ENGINEERING ----------
current_year = pd.Timestamp.now().year
if "yr_mfr" in df.columns:
    df["yr_mfr"] = pd.to_numeric(df["yr_mfr"], errors="coerce")
    df["age"] = current_year - df["yr_mfr"]
    df.loc[(df["age"] < 0) | (df["age"] > 100), "age"] = np.nan
else:
    df["age"] = np.nan

if "kms_run" in df.columns:
    df["kms_run"] = pd.to_numeric(df["kms_run"], errors="coerce")
    df.loc[df["kms_run"] < 0, "kms_run"] = np.nan
    df["kms_per_year"] = df.apply(lambda r: r["kms_run"] / r["age"] if pd.notna(r["kms_run"]) and pd.notna(r["age"]) and r["age"]>0 else np.nan, axis=1)
else:
    df["kms_per_year"] = np.nan

if "warranty_avail" in df.columns:
    df["warranty_avail"] = df["warranty_avail"].astype(str).str.lower().map({"true":1,"false":0,"yes":1,"no":0}).fillna(0).astype(int)

# Add engineered features
for f in ["age", "kms_per_year"]:
    if f not in features:
        features.append(f)

features = [f for f in features if f in df.columns]
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in CSV.")

df = df.dropna(subset=[target])
df = df[df[target] > 100]
df = df[df[target] < df[target].quantile(0.995)]

X = df[features].copy()
y = df[target].copy()

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---------- PIPELINE ----------
numeric_features = ["age", "kms_run", "kms_per_year", "original_price", "total_owners", "warranty_avail"]
numeric_features = [f for f in numeric_features if f in X_train.columns]
categorical_features = [f for f in X_train.columns if f not in numeric_features]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", **ohe_kwargs))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

# ---------- TRAIN ----------
print("Training model, please wait...")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ---------- EVALUATE ----------
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("\n Model trained successfully!")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# ---------- SAVE ----------
joblib.dump(pipeline, PIPE_PATH)
with open(FEATURE_JSON, "w") as f:
    json.dump({"features": features}, f)
print(f"\n Pipeline saved at: {PIPE_PATH}")
print(f" Feature list saved at: {FEATURE_JSON}")
