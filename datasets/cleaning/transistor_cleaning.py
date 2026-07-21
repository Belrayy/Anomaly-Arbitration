import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

input_file = "../data/unfit_data/transistor_unfit.csv"
output_file = "../data/unfit_data/transistor_cleaned.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

df.drop_duplicates(inplace=True)

missing_ratio = df.isnull().mean()
df = df.loc[:, missing_ratio < 0.5]

label_col = None
for col in ["Pass/Fail", "Label", "label", "Class"]:
    if col in df.columns:
        label_col = col
        break

feature_cols = [c for c in df.columns if c != label_col]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)

for col in feature_cols:
    df[col].fillna(df[col].median(), inplace=True)

constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]

df.drop(columns=constant_cols, inplace=True)

feature_cols = [c for c in df.columns if c != label_col]

corr = df[feature_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

df.drop(columns=to_drop, inplace=True)

feature_cols = [c for c in df.columns if c != label_col]

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

df.to_csv(output_file, index=False)

print("Final shape:", df.shape)
print("Removed constant columns:", len(constant_cols))
print("Removed correlated columns:", len(to_drop))
print("Saved as:", output_file)