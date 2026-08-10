import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDOneClassSVM

TRAIN_FILE = "../../../../datasets/data/split/cyber_train.csv"

MODEL_DIR = "../models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "sgd_one_class_svm_cyber.pkl"
)

NU = 0.05
RANDOM_STATE = 42
MAX_ITER = 1000
TOL = 1e-3

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
        "sgd_one_class_svm",
        SGDOneClassSVM(
            nu=NU,
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
            tol=TOL
        )
    )
])

print("Training SGD One-Class SVM...")

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
print(f"Nu            : {NU}")
print(f"Max iterations: {MAX_ITER}")
print(f"Tolerance     : {TOL}")
print(f"Model         : {MODEL_FILE}")