import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor

TRAIN_FILE = "../../../../datasets/data/split/cyber_train.csv"

MODEL_DIR = "../models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "local_outlier_factor_cyber.pkl"
)

N_NEIGHBORS = 20
CONTAMINATION = "auto"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

print("Loading training dataset...")

df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

print(f"Dataset shape: {df.shape}")

if "Label" not in df.columns:
    raise ValueError(
        "The CSV must contain a 'Label' column."
    )

X = df.drop(
    columns=["Label"]
)

X = X.select_dtypes(
    include=["number"]
)

X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

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
        "local_outlier_factor",
        LocalOutlierFactor(
            n_neighbors=N_NEIGHBORS,
            contamination=CONTAMINATION,
            novelty=True,
            n_jobs=-1
        )
    )
])

print("Training Local Outlier Factor...")

pipeline.fit(X)

print("Training completed.")

joblib.dump(
    pipeline,
    MODEL_FILE
)

print(
    f"Model saved to: {MODEL_FILE}"
)

print("\nTraining Summary")
print("-----------------------------")
print(f"Training file : {TRAIN_FILE}")
print(f"Samples       : {len(X)}")
print(f"Features      : {X.shape[1]}")
print(f"Neighbors     : {N_NEIGHBORS}")
print(f"Contamination : {CONTAMINATION}")
print(f"Model         : {MODEL_FILE}")