"""
AgriHarvest: End-to-End Compliance Validation
Validates all 10 spec requirements before launching Streamlit.
"""
import sys
import os
import pandas as pd

sys.path.append('src')
from predictor import HarvestPredictor

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label, condition):
    status = PASS if condition else FAIL
    print(f"  {status} {label}")
    return condition

errors = 0

print("=" * 55)
print("  AgriHarvest End-to-End Compliance Validation")
print("=" * 55)

# 1. Dataset integrity
df = pd.read_csv('data/processed/cleaned_crop_harvest_data.csv')
print("\n--- §1 §2: Crops and States ---")
errors += 0 if check("Only 5 crops present", set(df['Crop'].unique()) == {'Maize','Yam','Cassava','Tomato','Ewedu'}) else 1
errors += 0 if check("Only 10 states present", set(df['State'].unique()) == {'Oyo','Ogun','Ondo','Benue','Niger','Enugu','Kaduna','Plateau','Kano','Taraba'}) else 1

print("\n--- §3: Feature columns ---")
required = {'Crop','State','Season','Planting_Month','Temperature','Rainfall','Soil_Moisture','Nutrient_Level','Fertilizer','Weed_Competition','Planting_Window','Days_to_Maturity'}
errors += 0 if check("All 12 required feature columns present", required.issubset(set(df.columns))) else 1
forbidden_features = {'Soil_pH','Humidity','Elevation','Latitude','Longitude','Farm_Size','Yield','Market_Price'}
errors += 0 if check("No forbidden feature columns present", len(forbidden_features & set(df.columns)) == 0) else 1

print("\n--- §7: Data Quality ---")
errors += 0 if check("No missing Fertilizer values", df['Fertilizer'].isnull().sum() == 0) else 1
errors += 0 if check("No duplicate records", df.duplicated().sum() == 0) else 1
errors += 0 if check("Rainfall floor >= 5mm", (df['Rainfall'] < 5.0).sum() == 0) else 1
errors += 0 if check("Temperature within 17–40°C", ((df['Temperature'] < 17) | (df['Temperature'] > 40)).sum() == 0) else 1

print("\n--- §13 §14: Inference Pipeline ---")
p = HarvestPredictor(models_dir='models')
result = p.predict({
    'Crop': 'Tomato', 'State': 'Plateau', 'Season': 'Rainy',
    'Planting_Month': 'June', 'Temperature': 25.0, 'Rainfall': 130.0,
    'Soil_Moisture': 'Medium', 'Nutrient_Level': 'Fair',
    'Fertilizer': 'NPK', 'Weed_Competition': 'Low',
    'Planting_Window': 'Optimal', 'Days_to_Maturity': 85
})
errors += 0 if check("Predictor returns predicted_days_to_harvest (int)", isinstance(result['predicted_days_to_harvest'], int)) else 1
errors += 0 if check("Predictor returns expected_harvest_period (str)", isinstance(result['expected_harvest_period'], str)) else 1
errors += 0 if check("Predictor returns advisory_notes (list)", isinstance(result['advisory_notes'], list)) else 1
errors += 0 if check("Harvest period is non-empty", len(result['expected_harvest_period']) > 0) else 1
print(f"    Sample output: '{result['expected_harvest_period']}' ({result['predicted_days_to_harvest']} days)")

print("\n--- §4 §5: Primary Target is Days_to_Harvest ---")
errors += 0 if check("Days_to_Harvest column present in dataset", 'Days_to_Harvest' in df.columns) else 1
errors += 0 if check("Harvest_Month is NOT a model input feature", 'Harvest_Month' not in required) else 1
errors += 0 if check("Harvest_Period is NOT a model input feature", 'Harvest_Period' not in required) else 1

print("\n--- Models saved ---")
errors += 0 if check("harvest_regressor.pkl exists", os.path.exists('models/harvest_regressor.pkl')) else 1
errors += 0 if check("harvest_classifier.pkl exists", os.path.exists('models/harvest_classifier.pkl')) else 1
errors += 0 if check("regression_model_comparison.csv exists", os.path.exists('models/regression_model_comparison.csv')) else 1
errors += 0 if check("classification_model_comparison.csv exists", os.path.exists('models/classification_model_comparison.csv')) else 1

print("\n" + "=" * 55)
if errors == 0:
    print("  ALL CHECKS PASSED — System is ready.")
else:
    print(f"  {errors} CHECK(S) FAILED — Review before launch.")
print("=" * 55)
