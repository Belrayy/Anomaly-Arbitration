import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor

# Use the cleaned file (already one-hot encoded, no target column)
TRAIN_FILE = "../../../../datasets/data/split/creditcard_train.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "local_outlier_factor.pkl")

RANDOM_STATE = 42
CONTAMINATION = "auto"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE)

print(f"Samples: {len(df):,}")

# All columns are features
X = df

print(f"Features shape: {X.shape}")

# Scale all features
preprocessor = StandardScaler()

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=CONTAMINATION,
    novelty=True
)

pipeline = Pipeline([
    ("scaler", preprocessor),
    ("model", lof)
])

print("Training Local Outlier Factor...")

pipeline.fit(X)

print("Training completed.")

joblib.dump(pipeline, MODEL_FILE)

print(f"\nModel saved to:\n{MODEL_FILE}")

print("\nTraining Summary")
print("-----------------------------")
print("Algorithm      : Local Outlier Factor")
print(f"Neighbors      : {lof.n_neighbors}")
print(f"Contamination  : {CONTAMINATION}")
print(f"Features Used  : {len(X.columns)}")

print("\nFeatures:")
for feature in X.columns:
    print(f" - {feature}")

print("\nDone.")