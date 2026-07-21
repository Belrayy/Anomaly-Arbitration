import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

input_file = "../data/unfit_data/cyber_unfit.csv"
output_file = "../data/unfit_data/cyber_cleaned.csv"

df = pd.read_csv(input_file, low_memory=False)

print("Original shape:", df.shape)

df.columns = df.columns.str.strip()

df.drop_duplicates(inplace=True)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

label_col = "Label"

feature_cols = [c for c in df.columns if c != label_col]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(inplace=True)

constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]

df.drop(columns=constant_cols, inplace=True)

feature_cols = [c for c in df.columns if c != label_col]

scaler = StandardScaler()

df[feature_cols] = scaler.fit_transform(df[feature_cols])

df.reset_index(drop=True, inplace=True)

df.to_csv(output_file, index=False)

print("Final shape:", df.shape)
print("Removed constant columns:", len(constant_cols))
print("Saved as:", output_file)