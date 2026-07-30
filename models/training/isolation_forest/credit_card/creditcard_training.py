import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest


# Use the cleaned file (already one-hot encoded, no target column)
TRAIN_FILE = "../../../datasets/data/split/creditcard_train.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(MODEL_DIR, "isolation_forest.pkl")

RANDOM_STATE = 42

CONTAMINATION = "auto"

os.makedirs(MODEL_DIR, exist_ok=True)


print("Loading training dataset...")

df = pd.read_csv(TRAIN_FILE)

print(f"Samples: {len(df):,}")

# All columns are features - data is already one-hot encoded
X = df

print(f"Features shape: {X.shape}")

# Identify feature types
# All type columns are already one-hot encoded (0/1 integers)
# We still want to scale the continuous numerical features
one_hot_columns = [col for col in X.columns if col.startswith("type_")]
numerical_columns = [col for col in X.columns if col not in one_hot_columns]

print(f"One-hot encoded columns: {one_hot_columns}")
print(f"Numerical columns: {numerical_columns}")

# Since type is already one-hot encoded, we only need StandardScaler
# for the continuous numerical features
preprocessor = StandardScaler()


isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


pipeline = Pipeline([
    ("scaler", preprocessor),
    ("model", isolation_forest)
])


print("Training Isolation Forest...")

pipeline.fit(X)

print("Training completed.")


joblib.dump(pipeline, MODEL_FILE)

print(f"\nModel saved to:\n{MODEL_FILE}")

print("\nTraining Summary")
print("-----------------------------")
print("Algorithm      : Isolation Forest")
print(f"Trees          : {isolation_forest.n_estimators}")
print(f"Random State   : {RANDOM_STATE}")
print(f"Contamination  : {CONTAMINATION}")
print(f"Features Used  : {len(X.columns)}")

print("\nFeatures:")
for feature in X.columns:
    print(f" - {feature}")

print("\nDone.")