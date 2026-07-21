import pandas as pd

input_file = "../data/split/creditcard_train.csv"
output_file = "../data/split/creditcard_train.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

df.drop(columns=["nameOrig", "nameDest", "isFraud" , "isFlaggedFraud"], inplace=True) ## df.drop(columns=["nameOrig", "nameDest", "isFraud" , "isFlaggedFraud"], inplace=True)

df = pd.get_dummies(df, columns=["type"], dtype=int)

df.reset_index(drop=True, inplace=True)

df.to_csv(output_file, index=False)

print("Cleaned shape:", df.shape)
print(f"Saved as {output_file}")