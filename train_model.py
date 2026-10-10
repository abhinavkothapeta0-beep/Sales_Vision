from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing {DATA_PATH}. Download the Kaggle train.csv and place it in the data folder."
    )

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Exact feature engineering used in the uploaded notebook.
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

# Exact encoding used in the notebook.
df_encoded = pd.get_dummies(df, columns=["family"], drop_first=True)

X = df_encoded.drop(["sales", "date", "id"], axis=1)
y = df_encoded["sales"]

split_index = int(len(df_encoded) * 0.8)
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

print("Training shape:", X_train.shape)

rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

joblib.dump(rf_model, MODEL_DIR / "rf_model.joblib")
(MODEL_DIR / "feature_columns.json").write_text(
    json.dumps(list(X.columns), indent=2)
)

print("Saved:", MODEL_DIR / "rf_model.joblib")
print("Saved:", MODEL_DIR / "feature_columns.json")
print("Feature count:", len(X.columns))
print("Model ready for Streamlit.")
