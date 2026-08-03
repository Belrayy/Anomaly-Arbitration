import os
import pandas as pd
from sklearn.model_selection import train_test_split



INPUT_CSV = "../data/raw/transistor.csv"          
OUTPUT_DIR = "../data/split"

TRAIN_RATIO = 0.70
TEST_RATIO = 0.30
INFERENCE_RATIO = 0.50

RANDOM_STATE = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_CSV)

print(f"Original shape: {df.shape}")



if "Time" in df.columns:
    df = df.drop(columns=["Time"])
    print("Removed Time column.")



if "Pass/Fail" not in df.columns:
    raise ValueError("Pass/Fail column not found.")

y = df["Pass/Fail"]
X = df.drop(columns=["Pass/Fail"])



missing_percentage = X.isnull().mean()

cols_to_keep = missing_percentage[missing_percentage <= 0.50].index

X = X[cols_to_keep]

print(f"Remaining features: {X.shape[1]}")



X = X.fillna(X.median())

print("Missing values filled with median.")


cleaned_df = X.copy()
cleaned_df["Pass/Fail"] = y



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_RATIO,
    random_state=RANDOM_STATE,
    stratify=y
)

train_df = X_train.copy()
train_df["Pass/Fail"] = y_train

test_df = X_test.copy()
test_df["Pass/Fail"] = y_test


inference_df = X.sample(
    frac=INFERENCE_RATIO,
    random_state=RANDOM_STATE
).reset_index(drop=True)

train_path = os.path.join(OUTPUT_DIR, "transistor_train.csv")
test_path = os.path.join(OUTPUT_DIR, "transistor_test.csv")
inference_path = os.path.join(OUTPUT_DIR, "transistor_inference.csv")
#cleaned_path = os.path.join(OUTPUT_DIR, "transistor_cleaned.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
inference_df.to_csv(inference_path, index=False)
#cleaned_df.to_csv(cleaned_path, index=False)
