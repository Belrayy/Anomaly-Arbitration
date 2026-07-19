import pandas as pd
from sklearn.preprocessing import StandardScaler

input_file = "school.csv"
output_file = "school_cleaned.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

label_col = "final_result"

id_cols = ["id_student"]

categorical_cols = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

constant_cols = [
    c for c in df.columns
    if c != label_col and df[c].nunique() <= 1
]

df.drop(columns=constant_cols, inplace=True)

feature_cols = [
    c for c in df.columns
    if c not in [label_col] + id_cols
]

scaler = StandardScaler()

df[feature_cols] = scaler.fit_transform(df[feature_cols])

df["final_result"] = df["final_result"].replace({
    "Pass": 0,
    "Distinction": 0,
    "Fail": 1,
    "Withdrawn": 1
})

df.to_csv(output_file, index=False)

print("Final shape:", df.shape)
print(f"Removed constant columns: {len(constant_cols)}")
print(f"Saved as {output_file}")