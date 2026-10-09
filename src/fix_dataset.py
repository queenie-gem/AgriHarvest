import pandas as pd

for path in ['data/raw/nigerian_crop_harvest_data.csv', 'data/processed/cleaned_crop_harvest_data.csv']:
    df = pd.read_csv(path)

    # Fix 1: Fertilizer NaN => canonical label 'No Fertilizer'
    missing_before = int(df['Fertilizer'].isnull().sum())
    df['Fertilizer'] = df['Fertilizer'].fillna('No Fertilizer')
    missing_after = int(df['Fertilizer'].isnull().sum())
    print(f"{path}")
    print(f"  Fertilizer NaN fixed: {missing_before} -> {missing_after}")

    # Fix 2: Rainfall floor at 5.0 mm
    low_before = int((df['Rainfall'] < 5.0).sum())
    df['Rainfall'] = df['Rainfall'].clip(lower=5.0)
    low_after = int((df['Rainfall'] < 5.0).sum())
    print(f"  Rainfall < 5mm clamped: {low_before} -> {low_after}")

    df.to_csv(path, index=False)
    print("  Saved.")
    print()

print("Dataset fixes complete.")
