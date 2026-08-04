import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor

TRAIN_FILE = "../../../../datasets/data/split/transistor_train.csv"

MODEL_DIR = "../models"
MODEL_FILE = os.path.join(MODEL_DIR, "local_outlier_factor_transistor.pkl")

RANDOM_STATE = 42
CONTAMINATION = "auto"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE)

print(f"Samples: {len(df):,}")

if "Pass/Fail" in df.columns:
    X = df.drop(columns=["Pass/Fail"])
else:
    X = df.copy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

print(f"Features shape: {X.shape}")

preprocessor = StandardScaler()

lof = LocalOutlierFactor(
    n_neighbors=50,
    contamination="auto",
    novelty=True
)

pipeline = Pipeline([
    ("scaler", preprocessor),
    ("model", lof)
])

print("Training Local Outlier Factor...")

pipeline.fit(X)

print("Training completed.")

joblib.dump(
    {
        "pipeline": pipeline,
        "features": list(X.columns)
    },
    MODEL_FILE
)

print(f"\nModel saved to:\n{MODEL_FILE}")

print("\nTraining Summary")
print("-----------------------------")
print("Algorithm      : Local Outlier Factor")
print(f"Neighbors      : {lof.n_neighbors}")
print(f"Contamination  : {lof.contamination}")
print(f"Features Used  : {len(X.columns)}")

print("\nFeatures:")
for feature in X.columns:
    print(f" - {feature}")

print("\nDone.")