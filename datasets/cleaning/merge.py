import os
import glob
import pandas as pd

input_folder = "../data/cyber"
output_file = "merged_dataset.csv"

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

dataframes = []

for file in csv_files:
    df = pd.read_csv(file, low_memory=False)
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)

merged_df.to_csv(output_file, index=False)

print(f"Merged {len(csv_files)} files into '{output_file}'")
print(f"Total rows: {len(merged_df)}")