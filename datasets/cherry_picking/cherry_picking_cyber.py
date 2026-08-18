import os
import pandas as pd
import numpy as np

INPUT_FILE = "../data/raw/cyber.csv"
OUTPUT_FILE = "../data/cherry/cyber_200.csv"

RANDOM_STATE = 42

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Original dataset shape: {df.shape}")

print("\nCleaning column names...")
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

empty_columns = df.columns[df.isna().all()].tolist()
if empty_columns:
    print(f"\nRemoving {len(empty_columns)} completely empty columns:")
    print(empty_columns)
    df.drop(columns=empty_columns, inplace=True)

if "Label" in df.columns:
    print("\nCleaning Label column...")
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].replace({
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "None": np.nan,
        "null": np.nan,
        "NULL": np.nan
    })
    before = len(df)
    df = df.dropna(subset=["Label"])
    removed = before - len(df)
    if removed > 0:
        print(f"Removed {removed} rows with missing Label.")

print("\nConverting feature columns to numeric...")
feature_columns = [column for column in df.columns if column != "Label"]
for column in feature_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nReplacing infinite values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nRemoving duplicate rows...")
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df)}")

print("\nHandling missing numeric values...")
numeric_columns = df.select_dtypes(include=[np.number]).columns
missing_before = df[numeric_columns].isna().sum().sum()
print(f"Missing numeric values before cleaning: {missing_before}")

for column in numeric_columns:
    if df[column].isna().any():
        median_value = df[column].median()
        if pd.isna(median_value):
            median_value = 0
        df[column] = df[column].fillna(median_value)

missing_after = df[numeric_columns].isna().sum().sum()
print(f"Missing numeric values after cleaning: {missing_after}")

print("\nChecking for constant features...")
constant_columns = []
for column in numeric_columns:
    if df[column].nunique(dropna=False) <= 1:
        constant_columns.append(column)

if constant_columns:
    print(f"Removing {len(constant_columns)} constant columns:")
    for column in constant_columns:
        print(f"  - {column}")
    df.drop(columns=constant_columns, inplace=True)
else:
    print("No constant features found.")

print("\nFinal cleaning check...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

numeric_columns = df.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    if df[column].isna().any():
        median_value = df[column].median()
        if pd.isna(median_value):
            median_value = 0
        df[column] = df[column].fillna(median_value)

print(f"Cleaned dataset shape: {df.shape}")

sample_size = min(200, len(df))
df = df.sample(n=sample_size, random_state=RANDOM_STATE).copy()
print(f"Random sample shape: {df.shape}")

if "Label" in df.columns:
    print("\nLabel removed from final inference-style output.")
    df = df.drop(columns=["Label"])

output_dir = os.path.dirname(OUTPUT_FILE)
os.makedirs(output_dir, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
print(f"Original rows : {len(df):,}")
print(f"Final sample  : {len(df):,}")
print(f"Columns       : {len(df.columns)}")

print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"\nGenerated file: {OUTPUT_FILE}")
print("\nDone!")