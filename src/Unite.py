import pandas as pd
from sklearn.model_selection import train_test_split

politifact_df = pd.read_csv('../data/politifact_clean.csv')
gossipcop_df = pd.read_csv('../data/gossipcop_clean.csv')

combined_df = pd.concat([politifact_df, gossipcop_df], ignore_index=True)

print(f"✅ Об’єднано {len(combined_df)} записів")

combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_val_df, test_df = train_test_split(combined_df, test_size=0.15, random_state=42, stratify=combined_df['binary_label'])

train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42, stratify=train_val_df['binary_label'])

print(f"✅ Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")


train_df.to_csv('../data/combined_train.csv', index=False)
val_df.to_csv('../data/combined_val.csv', index=False)
test_df.to_csv('../data/combined_test.csv', index=False)

print("✅ Saved train, val, and test splits")
