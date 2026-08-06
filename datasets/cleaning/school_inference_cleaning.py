import os
import joblib
import pandas as pd

INPUT_FILE = "../data/split/school_inference.csv"

MODEL_FILE = "../../models/training/isolation_forest/models/isolation_forest_school.pkl"

OUTPUT_FILE = "../data/split/school_inference_processed.csv"


print("Loading inference dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df)}")

columns_to_drop = [
    "id_student",
    "final_result",
    "date_registration",
    "date_unregistration"
]

existing = [c for c in columns_to_drop if c in df.columns]

if existing:
    df = df.drop(columns=existing)

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Encoding categorical columns...")
print(categorical_columns)

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=False,
    dtype=int
)

saved = joblib.load(MODEL_FILE)

expected_features = saved["features"]

missing = [c for c in expected_features if c not in df.columns]

for column in missing:
    df[column] = 0

extra = [c for c in df.columns if c not in expected_features]

if extra:
    print("Removing unexpected columns:")
    print(extra)

df = df.drop(columns=extra)
df = df[expected_features]
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.median(numeric_only=True))

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone!")
print(f"Processed inference dataset saved to:\n{OUTPUT_FILE}")
print(f"Rows     : {len(df)}")
print(f"Features : {len(df.columns)}")