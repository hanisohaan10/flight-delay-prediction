import glob
import pandas as pd

files = sorted(glob.glob("data/raw/*.csv"))
print(files)

dfs = [pd.read_csv(f, low_memory=False) for f in files]

# Verify columns match
cols = [tuple(d.columns) for d in dfs]
assert all(c == cols[0] for c in cols), "Columns don't match across files!"

# CONCAT FIRST
df = pd.concat(dfs, ignore_index=True)

# Then drop phantom column
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# Then save
df.to_parquet("data/processed/flights.parquet")

# Then inspect
print(df.shape)
print(df.columns.tolist())
df.head()