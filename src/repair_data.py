"""
One-time data repair: fills missing Fertilizer values with 'No Fertilizer' and clamps Rainfall floor to 5.0mm.
Saves directly to both raw and processed CSVs.
"""
import pandas as pd

paths = [
    'data/raw/nigerian_crop_harvest_data.csv',
    'data/processed/cleaned_crop_harvest_data.csv'
]

for path in paths:
    df = pd.read_csv(path)
    before = int(df['Fertilizer'].isnull().sum())
    df['Fertilizer'] = df['Fertilizer'].fillna('No Fertilizer')
    df['Rainfall'] = df['Rainfall'].clip(lower=5.0)
    df.to_csv(path, index=False)
    after = int(df['Fertilizer'].isnull().sum())
    print(f"{path}: Fertilizer NaN {before} -> {after} | saved.")

print("Done.")
