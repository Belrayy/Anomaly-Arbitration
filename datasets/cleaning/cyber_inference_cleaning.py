import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


INPUT_FILE = "../data/raw/cyber.csv"
OUTPUT_DIR = "../data/split"

TRAIN_FILE = os.path.join(OUTPUT_DIR, "cyber_train.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "cyber_test.csv")
INFERENCE_FILE = os.path.join(OUTPUT_DIR, "cyber_inference.csv")

TRAIN_SIZE = 0.70
TEST_SIZE = 0.30
INFERENCE_SIZE = 0.50

RANDOM_STATE = 42


os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    # Convert common empty values to NaN
    df["Label"] = df["Label"].replace(
        {
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "null": np.nan,
            "NULL": np.nan
        }
    )

    # Remove rows without a label
    before = len(df)

    df = df.dropna(subset=["Label"])

    removed = before - len(df)

    if removed > 0:
        print(f"Removed {removed} rows with missing Label.")


print("\nConverting feature columns to numeric...")

feature_columns = [
    column for column in df.columns
    if column != "Label"
]

for column in feature_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


print("\nReplacing infinite values...")

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


print("\nRemoving duplicate rows...")

before = len(df)

df.drop_duplicates(inplace=True)

duplicates_removed = before - len(df)

print(f"Duplicates removed: {duplicates_removed}")



print("\nHandling missing numeric values...")

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

missing_before = df[numeric_columns].isna().sum().sum()

print(f"Missing numeric values before cleaning: {missing_before}")

for column in numeric_columns:

    if df[column].isna().any():

        median_value = df[column].median()

        # If the whole column is NaN, use 0
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

    print(
        f"Removing {len(constant_columns)} constant columns:"
    )

    for column in constant_columns:
        print(f"  - {column}")

    df.drop(columns=constant_columns, inplace=True)

else:

    print("No constant features found.")


print("\nFinal cleaning check...")

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

for column in numeric_columns:

    if df[column].isna().any():

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0

        df[column] = df[column].fillna(median_value)


print(f"Cleaned dataset shape: {df.shape}")

print("\nSaving cleaned dataset...")

df.to_csv(
    INFERENCE_FILE,
    index=False
)

print(f"Cleaned dataset saved to:")
print(INFERENCE_FILE)


print("\n" + "=" * 70)
print("CREATING TRAIN / TEST DATASETS")
print("=" * 70)

if "Label" in df.columns:

    train_df, test_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Label"]
    )

else:

    train_df, test_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

print(f"\nTrain set: {len(train_df):,} rows")
print(f"Test set : {len(test_df):,} rows")

print(f"\nTrain set saved to:")
print(TRAIN_FILE)

print(f"\nTest set saved to:")
print(TEST_FILE)


print("\n" + "=" * 70)
print("CREATING INFERENCE DATASET")
print("=" * 70)

inference_df = df.sample(
    frac=INFERENCE_SIZE,
    random_state=RANDOM_STATE
).copy()


if "Label" in inference_df.columns:

    

    inference_df.drop(
        columns=["Label"],
        inplace=True
    )

    print("\nLabel removed from inference dataset.")

    print("True inference labels saved separately to:")




print(f"\nInference set: {len(inference_df):,} rows")

print("\nInference set saved to:")

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Original rows : {len(df):,}")
print(f"Train rows    : {len(train_df):,}")
print(f"Test rows     : {len(test_df):,}")
print(f"Inference rows: {len(inference_df):,}")

print(f"\nNumber of features: {len(inference_df.columns)}")

if "Label" in df.columns:

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    label_distribution = (
        df["Label"]
        .value_counts()
        .to_frame("Count")
    )

    label_distribution["Percentage"] = (
        label_distribution["Count"]
        / len(df)
        * 100
    )

    print(label_distribution)


print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(f"1. {INFERENCE_FILE}")
print(f"2. {TRAIN_FILE}")
print(f"3. {TEST_FILE}")

print("\nDone!")