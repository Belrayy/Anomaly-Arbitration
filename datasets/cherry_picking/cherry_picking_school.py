import os
import pandas as pd

INPUT_FILE = "../data/raw/school.csv"
OUTPUT_FILE = "../data/cherry/school_200.csv"

RANDOM_STATE = 42

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

sample_size = min(200, len(df))
df = df.sample(n=sample_size, random_state=RANDOM_STATE).copy()

print("Sampled shape:", df.shape)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

columns_to_drop = [
    "id_student",
    "final_result",
    "date_registration",
    "date_unregistration"
]

existing = [c for c in columns_to_drop if c in df.columns]
df = df.drop(columns=existing)

categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

df = pd.get_dummies(df, columns=categorical_columns, dtype=int)

df.reset_index(drop=True, inplace=True)

df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned shape:", df.shape)
print(f"Saved as {OUTPUT_FILE}")