import pandas as pd

input_file = "../data/split/creditcard_test.csv"
output_file = "../data/split/creditcard_inference.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df.drop(columns=["isFraud"], inplace=True)   

df.reset_index(drop=True, inplace=True)

df.to_csv(output_file, index=False)

print("Cleaned shape:", df.shape)
print(f"Saved as {output_file}")