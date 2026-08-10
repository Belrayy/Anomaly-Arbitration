import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

TRAIN_FILE = "../../../../datasets/data/split/cyber_train.csv"

MODEL_DIR = "../models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "isolation_forest_cyber.pkl"
)

RANDOM_STATE = 42
N_ESTIMATORS = 300
CONTAMINATION = "auto"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE)

print(f"Dataset shape: {df.shape}")

if "Label" not in df.columns:
    raise ValueError("The CSV must contain a 'Label' column.")

X = df.drop(columns=["Label"])

X = X.select_dtypes(include=["number"])

X = X.replace([float("inf"), float("-inf")], pd.NA)

before = len(X)

X = X.dropna()

after = len(X)

print(f"Rows removed: {before - after}")
print(f"Training samples: {len(X)}")
print(f"Features: {X.shape[1]}")

pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "isolation_forest",
        IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])

print("Training Isolation Forest...")

pipeline.fit(X)

print("Training completed.")

joblib.dump(pipeline, MODEL_FILE)

print(f"Model saved to: {MODEL_FILE}")