import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest


TRAIN_FILE = "../../../../datasets/data/split/transistor_train.csv"

MODEL_DIR = "../models"
MODEL_FILE = os.path.join(MODEL_DIR, "isolation_forest_transistor.pkl")

RANDOM_STATE = 42
CONTAMINATION = "auto"
N_ESTIMATORS = 300

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE)

print(f"Samples: {len(df):,}")

LABEL_COLUMNS = [
    "Pass/Fail",
    "Class",
    "Label",
    "label",
    "class",
    "isFraud",
    "target"
]

label_found = None

for column in LABEL_COLUMNS:
    if column in df.columns:
        label_found = column
        break

if label_found is not None:
    print(f"Removing label column: {label_found}")
    X = df.drop(columns=[label_found])
else:
    print("No label column detected.")
    X = df.copy()

print(f"Feature matrix shape: {X.shape}")


X = X.apply(pd.to_numeric, errors="coerce")

X = X.fillna(X.median())

print("All features converted to numeric.")


pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            max_samples="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


print("\nTraining Isolation Forest...")

pipeline.fit(X)

print("Training completed.")

scaled_X = pipeline.named_steps["scaler"].transform(X)

scores = pipeline.named_steps["model"].score_samples(scaled_X)

print("\nTraining anomaly score statistics")
print("---------------------------------")
print(f"Mean score : {scores.mean():.6f}")
print(f"Min score  : {scores.min():.6f}")
print(f"Max score  : {scores.max():.6f}")


joblib.dump(
    {
        "pipeline": pipeline,
        "features": list(X.columns)
    },
    MODEL_FILE
)

print(f"\nModel saved to:\n{MODEL_FILE}")

print("\n==============================")
print("TRAINING SUMMARY")
print("==============================")
print("Algorithm        : Isolation Forest")
print(f"Samples          : {len(X):,}")
print(f"Features         : {X.shape[1]}")
print(f"Trees            : {N_ESTIMATORS}")
print(f"Contamination    : {CONTAMINATION}")
print(f"Random State     : {RANDOM_STATE}")

print("\nFeature names:")
for feature in X.columns:
    print(f" - {feature}")

print("\nDone.")