import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "../data/school_dataset.csv"
train_file = "../data/clean/train.csv"
test_file = "../data/clean/test.csv"

df = pd.read_csv(input_file)

train_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    shuffle=True
)

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"Training set: {len(train_df)} rows")
print(f"Test set: {len(test_df)} rows")