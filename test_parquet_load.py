import pandas as pd
import os

parquet_path = "data/curated_trips"
print(f"Testing parquet path: {parquet_path}")
print(f"Path exists: {os.path.exists(parquet_path)}")

if os.path.exists(parquet_path):
    try:
        print("Attempting pd.read_parquet()...")
        df = pd.read_parquet(parquet_path)
        print(f"✓ SUCCESS! Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print("Data source would be: PARQUET ✓")
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        print("Data source would be: CSV")
