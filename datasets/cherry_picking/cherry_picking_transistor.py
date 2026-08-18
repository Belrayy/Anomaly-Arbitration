import os
import pandas as pd

INPUT_CSV = "../data/raw/transistor.csv"
OUTPUT_FILE = "../data/cherry/transistor_200.csv"

RANDOM_STATE = 42

print("Loading dataset...")

df = pd.read_csv(INPUT_CSV)
print(f"Original shape: {df.shape}")

if "Time" in df.columns:
    df = df.drop(columns=["Time"])
    print("Removed Time column.")

if "Pass/Fail" not in df.columns:
    raise ValueError("Pass/Fail column not found.")

# Keep the same preprocessing behavior, but just on a random 200-row sample
sample_size = min(200, len(df))
df = df.sample(n=sample_size, random_state=RANDOM_STATE).copy()
print(f"Sampled shape: {df.shape}")

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

cleaned_df = cleaned_df.reset_index(drop=True)
cleaned_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved inference-style sample to: {OUTPUT_FILE}")
print(f"Final shape: {cleaned_df.shape}")
