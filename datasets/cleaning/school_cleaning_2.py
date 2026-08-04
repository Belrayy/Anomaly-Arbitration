import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "../data/raw/school.csv"
OUTPUT_DIR = "../data/split"

RANDOM_STATE = 42
TEST_SIZE = 0.30
INFERENCE_FRACTION = 0.50

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("Creating inference dataset...")

df_inference = df.sample(
    frac=INFERENCE_FRACTION,
    random_state=RANDOM_STATE
)

df_inference.to_csv(
    os.path.join(OUTPUT_DIR, "school_inference.csv"),
    index=False
)

target = None

if "final_result" in df.columns:
    target = df["final_result"]

columns_to_drop = [
    "id_student",
    "final_result",
    "date_registration",
    "date_unregistration"
]

existing = [c for c in columns_to_drop if c in df.columns]

df = df.drop(columns=existing)

print("\nRemaining columns:")
print(df.columns.tolist())

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=False,
    dtype=int
)
print("\nScaling...")

scaler = StandardScaler()

scaled = scaler.fit_transform(df)

X = pd.DataFrame(
    scaled,
    columns=df.columns
)

print("\nSplitting dataset...")

if target is not None:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target
    )

    train = X_train.copy()
    train["final_result"] = y_train.values

    test = X_test.copy()
    test["final_result"] = y_test.values

else:

    X_train, X_test = train_test_split(
        X,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train = X_train
    test = X_test

train_file = os.path.join(
    OUTPUT_DIR,
    "school_train.csv"
)

test_file = os.path.join(
    OUTPUT_DIR,
    "school_test.csv"
)



train.to_csv(
    train_file,
    index=False
)

test.to_csv(
    test_file,
    index=False
)


print("Dataset preparation complete")

print(f"Original rows      : {len(df)}")
print(f"Processed features : {len(X.columns)}")
print(f"Training rows      : {len(train)}")
print(f"Testing rows       : {len(test)}")
print(f"Inference rows     : {len(df_inference)}")

print("\nSaved files:")

print(f"- {train_file}")
print(f"- {test_file}")
print(f"- {os.path.join(OUTPUT_DIR,'school_inference.csv')}")