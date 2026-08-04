import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDOneClassSVM

TRAIN_FILE = "../../../../datasets/data/split/transistor_train.csv"

MODEL_DIR = "../models"
MODEL_FILE = os.path.join(MODEL_DIR, "sgd_one_class_svm_transistor.pkl")

NU = 0.01
RANDOM_STATE = 42

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
    X = df.copy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

print(f"Features shape: {X.shape}")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SGDOneClassSVM(
        nu=NU,
        random_state=RANDOM_STATE,
        max_iter=10000,
        tol=1e-3
    ))
])

print("Training SGD One-Class SVM...")

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
print("Algorithm      : SGD One-Class SVM")
print(f"Nu             : {NU}")
print(f"Random State   : {RANDOM_STATE}")
print(f"Features Used  : {len(X.columns)}")

print("\nFeatures:")
for feature in X.columns:
    print(f" - {feature}")

print("\nDone.")