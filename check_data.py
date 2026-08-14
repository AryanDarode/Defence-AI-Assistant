import pandas as pd

file_path = "data/raw/DRDO.csv"

df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 Records:")
print(df[
    [
        "document_id",
        "title",
        "category",
        "local_path",
        "text_path",
        "content_type",
        "status"
    ]
].head(10).to_string(index=False))

print("\nUnique Content Types:")
print(df["content_type"].value_counts())

print("\nUnique Status:")
print(df["status"].value_counts())