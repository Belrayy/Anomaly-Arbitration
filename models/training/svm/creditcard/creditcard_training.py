import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDOneClassSVM

TRAIN_FILE = "../../../../datasets/data/split/creditcard_train.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sgd_one_class_svm.pkl")

NU = 0.01
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE)

print(f"Samples: {len(df):,}")

X = df

print(f"Features shape: {X.shape}")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SGDOneClassSVM(
        nu=NU,
        random_state=RANDOM_STATE,
        max_iter=5000,
        tol=1e-3
    ))
])

print("Training SGD One-Class SVM...")

pipeline.fit(X)

print("Training completed.")

joblib.dump(pipeline, MODEL_FILE)

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