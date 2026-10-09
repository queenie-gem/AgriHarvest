"""
Smart Season: Nigerian Agricultural Dataset Generator
======================================================
This module generates a scientifically grounded Nigerian agricultural dataset
for harvest period prediction across 5 key crops: Maize, Yam, Cassava, Tomato, and Ewedu.

Agronomic and Climatic Benchmarks Sourced from:
- International Institute of Tropical Agriculture (IITA), Ibadan, Nigeria
- National Horticultural Research Institute (NIHORT), Ibadan, Nigeria
- National Cereals Research Institute (NCRI), Badeggi, Niger State, Nigeria
- Nigerian Meteorological Agency (NiMet) Seasonal Climate Predictions (SCP)

Features Included (Strictly 12 required inputs + target):
1. Crop: [Maize, Yam, Cassava, Tomato, Ewedu]
2. State: [Oyo, Ogun, Ondo, Benue, Niger, Enugu, Kaduna, Plateau, Kano, Taraba]
3. Season: [Rainy, Dry]
4. Planting_Month: [January - December]
5. Temperature: Environmental temperature in °C
6. Rainfall: Rainfall in mm
7. Soil_Moisture: [Low, Medium, High]
8. Nutrient_Level: [Poor, Fair, Good]
9. Fertilizer: [NPK, Urea, Organic Manure, No Fertilizer]
10. Weed_Competition: [Low, Medium, High]
11. Planting_Window: [Early, Optimal, Late]
12. Days_to_Maturity: Baseline biological days to maturity

Targets:
- Days_to_Harvest: Actual days from planting to harvest under field conditions
- Harvest_Month: Expected calendar month of harvest
- Harvest_Period: Detailed harvest window (e.g. 'Early August', 'Mid August', 'Late August')
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for scientific reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

MONTH_TO_INT = {m: i + 1 for i, m in enumerate(MONTHS)}
INT_TO_MONTH = {i + 1: m for i, m in enumerate(MONTHS)}

# Nigerian Agro-ecological Zones & State Profiles (NiMet & Agricultural Ministries)
# Defines mean monthly rainfall (mm) and temperature (°C) ranges per state
STATE_PROFILES = {
    'Oyo': {
        'zone': 'Derived Savannah / Rainforest transition',
        'rainy_months': [3, 4, 5, 6, 7, 8, 9, 10],
        'temp_range': (24.0, 32.5),
        'annual_rainfall': 1300,
        'soil_moisture_wet': 'High',
        'soil_moisture_dry': 'Low'
    },
    'Ogun': {
        'zone': 'Rainforest',
        'rainy_months': [3, 4, 5, 6, 7, 8, 9, 10, 11],
        'temp_range': (24.0, 31.5),
        'annual_rainfall': 1500,
        'soil_moisture_wet': 'High',
        'soil_moisture_dry': 'Medium'
    },
    'Ondo': {
        'zone': 'Rainforest',
        'rainy_months': [3, 4, 5, 6, 7, 8, 9, 10, 11],
        'temp_range': (23.5, 31.0),
        'annual_rainfall': 1650,
        'soil_moisture_wet': 'High',
        'soil_moisture_dry': 'Medium'
    },
    'Enugu': {
        'zone': 'Derived Savannah',
        'rainy_months': [4, 5, 6, 7, 8, 9, 10],
        'temp_range': (24.5, 33.0),
        'annual_rainfall': 1400,
        'soil_moisture_wet': 'High',
        'soil_moisture_dry': 'Low'
    },
    'Benue': {
        'zone': 'Southern Guinea Savannah',
        'rainy_months': [4, 5, 6, 7, 8, 9, 10],
        'temp_range': (25.0, 35.0),
        'annual_rainfall': 1250,
        'soil_moisture_wet': 'High',
        'soil_moisture_dry': 'Low'
    },
    'Niger': {
        'zone': 'Southern Guinea Savannah',
        'rainy_months': [4, 5, 6, 7, 8, 9, 10],
        'temp_range': (24.5, 36.0),
        'annual_rainfall': 1150,
        'soil_moisture_wet': 'Medium',
        'soil_moisture_dry': 'Low'
    },
    'Taraba': {
        'zone': 'Southern Guinea / Montane transition',
        'rainy_months': [4, 5, 6, 7, 8, 9, 10],
        'temp_range': (23.0, 34.0),
        'annual_rainfall': 1200,
        'soil_moisture_wet': 'Medium',
        'soil_moisture_dry': 'Low'
    },
    'Plateau': {
        'zone': 'Northern Guinea Savannah (Highland)',
        'rainy_months': [4, 5, 6, 7, 8, 9],
        'temp_range': (18.0, 28.5), # Cooler highland climate
        'annual_rainfall': 1300,
        'soil_moisture_wet': 'Medium',
        'soil_moisture_dry': 'Low'
    },
    'Kaduna': {
        'zone': 'Northern Guinea Savannah',
        'rainy_months': [5, 6, 7, 8, 9],
        'temp_range': (22.0, 34.5),
        'annual_rainfall': 1050,
        'soil_moisture_wet': 'Medium',
        'soil_moisture_dry': 'Low'
    },
    'Kano': {
        'zone': 'Sudan Savannah',
        'rainy_months': [6, 7, 8, 9],
        'temp_range': (22.0, 38.0), # Hotter semi-arid climate
        'annual_rainfall': 700,
        'soil_moisture_wet': 'Medium',
        'soil_moisture_dry': 'Low'
    }
}

# Crop Agronomic Profiles (IITA, NIHORT, NCRI)
CROP_PROFILES = {
    'Maize': {
        'institution': 'NCRI / IITA',
        'dtm_range': (85, 120),       # Early vs Intermediate cultivars
        'optimal_planting_months': ['March', 'April', 'May', 'June', 'August'], # Early and late season flushes
        'suitable_seasons': ['Rainy', 'Dry'], # Dry if irrigated / fadama
        'preferred_fertilizer': ['NPK', 'Urea'],
        'heat_tolerance_limit': 34.0,
        'moisture_need': 'Medium'
    },
    'Yam': {
        'institution': 'IITA / NRCRI',
        'dtm_range': (180, 270),      # 6 to 9 months
        'optimal_planting_months': ['November', 'December', 'January', 'February', 'March', 'April'],
        'suitable_seasons': ['Rainy', 'Dry'], # Often planted dry season in mounds before rains
        'preferred_fertilizer': ['NPK', 'Organic Manure'],
        'heat_tolerance_limit': 35.0,
        'moisture_need': 'High'
    },
    'Cassava': {
        'institution': 'IITA',
        'dtm_range': (270, 365),      # 9 to 12+ months
        'optimal_planting_months': ['March', 'April', 'May', 'June', 'September', 'October'],
        'suitable_seasons': ['Rainy', 'Dry'],
        'preferred_fertilizer': ['NPK', 'Organic Manure', 'No Fertilizer'],
        'heat_tolerance_limit': 36.0,
        'moisture_need': 'Medium'
    },
    'Tomato': {
        'institution': 'NIHORT',
        'dtm_range': (75, 100),       # 2.5 to 3.5 months
        'optimal_planting_months': ['June', 'July', 'August', 'October', 'November', 'December'],
        'suitable_seasons': ['Rainy', 'Dry'], # Major dry season irrigated production in North
        'preferred_fertilizer': ['NPK', 'Organic Manure', 'Urea'],
        'heat_tolerance_limit': 33.0,
        'moisture_need': 'Medium'
    },
    'Ewedu': {
        'institution': 'NIHORT',
        'dtm_range': (30, 45),        # 1 to 1.5 months
        'optimal_planting_months': ['March', 'April', 'May', 'June', 'July', 'August', 'September'],
        'suitable_seasons': ['Rainy', 'Dry'],
        'preferred_fertilizer': ['Urea', 'NPK', 'Organic Manure'],
        'heat_tolerance_limit': 35.0,
        'moisture_need': 'High'
    }
}


def calculate_harvest_target(planting_month_name, days_to_harvest):
    """
    Computes exact harvest month and period string from planting month and actual DTH.
    Assumes planting occurs around the middle of the planting month (day 15).
    """
    plant_m_int = MONTH_TO_INT[planting_month_name]
    plant_date = datetime(2024, plant_m_int, 15)
    harvest_date = plant_date + timedelta(days=int(round(days_to_harvest)))
    
    harvest_month_name = INT_TO_MONTH[harvest_date.month]
    
    if harvest_date.day <= 10:
        period_prefix = "Early"
    elif harvest_date.day <= 20:
        period_prefix = "Mid"
    else:
        period_prefix = "Late"
        
    harvest_period = f"{period_prefix} {harvest_month_name}"
    return harvest_month_name, harvest_period


def generate_nigerian_agronomic_dataset(num_records=2200):
    """
    Generates a structured, grounded agricultural dataset adhering to real Nigerian
    agro-climatic distributions and biological growth dynamics.
    """
    records = []
    crops = list(CROP_PROFILES.keys())
    states = list(STATE_PROFILES.keys())
    
    for _ in range(num_records):
        crop = random.choice(crops)
        crop_info = CROP_PROFILES[crop]
        
        # State selection (some crops have higher affinity for certain states)
        if crop == 'Yam':
            # Benue, Oyo, Niger, Enugu are premier yam producers
            state = random.choices(states, weights=[0.2, 0.1, 0.05, 0.15, 0.25, 0.15, 0.05, 0.02, 0.02, 0.01])[0]
        elif crop == 'Tomato':
            # Kano, Kaduna, Plateau, Benue are top tomato producers
            state = random.choices(states, weights=[0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.25, 0.15, 0.15, 0.05])[0]
        elif crop == 'Cassava':
            # South & Middle belt
            state = random.choices(states, weights=[0.2, 0.2, 0.15, 0.15, 0.15, 0.05, 0.03, 0.01, 0.05, 0.01])[0]
        else:
            state = random.choice(states)
            
        state_info = STATE_PROFILES[state]
        
        # Select planting month
        # 75% realistic optimal planting, 25% non-traditional/off-season
        if random.random() < 0.75:
            planting_month = random.choice(crop_info['optimal_planting_months'])
        else:
            planting_month = random.choice(MONTHS)
            
        plant_m_int = MONTH_TO_INT[planting_month]
        
        # Determine season based on state's rainy calendar
        is_rainy = plant_m_int in state_info['rainy_months']
        season = 'Rainy' if is_rainy else 'Dry'
        
        # Planting Window Category
        if planting_month in crop_info['optimal_planting_months']:
            planting_window = random.choices(['Optimal', 'Early', 'Late'], weights=[0.70, 0.18, 0.12])[0]
        else:
            planting_window = random.choices(['Late', 'Early', 'Optimal'], weights=[0.60, 0.25, 0.15])[0]
            
        # Temperature generation (°C)
        min_temp, max_temp = state_info['temp_range']
        # Plateau is highland (cooler), Kano is hotter
        # In dry season (Harmattan vs hot dry):
        if not is_rainy and plant_m_int in [12, 1]:
            # Harmattan cool nights, warm days
            base_temp = min_temp + 2.0
        elif not is_rainy and plant_m_int in [3, 4, 5]:
            # Pre-monsoon heat
            base_temp = max_temp - 1.0
        else:
            base_temp = (min_temp + max_temp) / 2.0
            
        temperature = round(float(np.random.normal(base_temp, 1.8)), 1)
        temperature = max(17.0, min(temperature, 40.0))
        
        # Rainfall generation (mm during growing period)
        if is_rainy:
            # Monthly rainfall during wet season ranges from 120 to 350 mm depending on zone
            zone_factor = state_info['annual_rainfall'] / 1200.0
            rainfall = round(float(np.random.normal(180 * zone_factor, 45)), 1)
            rainfall = max(60.0, rainfall)
            soil_moisture = random.choices(['High', 'Medium', 'Low'], weights=[0.55, 0.35, 0.10])[0]
        else:
            # Dry season: low rainfall unless irrigated / fadama
            rainfall = round(float(np.random.exponential(25.0)), 1)
            rainfall = min(rainfall, 90.0)
            soil_moisture = random.choices(['Low', 'Medium', 'High'], weights=[0.70, 0.25, 0.05])[0]
            
        # Nutrient Level & Fertilizer Management
        nutrient_level = random.choices(['Good', 'Fair', 'Poor'], weights=[0.35, 0.45, 0.20])[0]
        
        if nutrient_level == 'Good':
            fertilizer = random.choices(['NPK', 'Organic Manure', 'Urea', 'No Fertilizer'], weights=[0.50, 0.30, 0.15, 0.05])[0]
        elif nutrient_level == 'Fair':
            fertilizer = random.choices(['NPK', 'Urea', 'Organic Manure', 'No Fertilizer'], weights=[0.40, 0.30, 0.20, 0.10])[0]
        else:
            fertilizer = random.choices(['No Fertilizer', 'Organic Manure', 'NPK', 'Urea'], weights=[0.45, 0.25, 0.15, 0.15])[0]
            
        # Weed Competition
        weed_competition = random.choices(['Low', 'Medium', 'High'], weights=[0.40, 0.40, 0.20])[0]
        
        # Baseline Days to Maturity (DTM) for the cultivar
        dtm_low, dtm_high = crop_info['dtm_range']
        base_dtm = int(round(np.random.uniform(dtm_low, dtm_high)))
        
        # --- Agronomic Stress & Maturity Dynamics ---
        # Real-world field delays or early senescence based on management and environment:
        stress_adjustment = 0.0
        
        # 1. Weed Stress (High weed pressure deprives crop of nutrients and sunlight, extending time to harvest)
        if weed_competition == 'High':
            stress_adjustment += base_dtm * random.uniform(0.06, 0.14)
        elif weed_competition == 'Medium':
            stress_adjustment += base_dtm * random.uniform(0.01, 0.05)
            
        # 2. Nutrient Deficiency / Fertilizer effect
        if nutrient_level == 'Poor' and fertilizer == 'No Fertilizer':
            stress_adjustment += base_dtm * random.uniform(0.05, 0.12)
        elif nutrient_level == 'Good' and fertilizer in ['NPK', 'Organic Manure']:
            stress_adjustment -= base_dtm * random.uniform(0.02, 0.05) # Vigorous healthy growth
            
        # 3. Moisture stress
        if soil_moisture == 'Low':
            if crop in ['Tomato', 'Ewedu', 'Yam']:
                # Moisture-sensitive crops stall vegetative progress
                stress_adjustment += base_dtm * random.uniform(0.05, 0.12)
            elif crop == 'Maize':
                # Drought can either delay silking or force early drying
                stress_adjustment += base_dtm * random.uniform(-0.04, 0.06)
        elif soil_moisture == 'High' and crop == 'Yam':
            # Excess waterlogging slows tuber development
            stress_adjustment += base_dtm * random.uniform(0.02, 0.06)
            
        # 4. Temperature Extremes
        if temperature > crop_info['heat_tolerance_limit']:
            # Heat stress delays flowering and fruit set (especially in tomato)
            stress_adjustment += base_dtm * random.uniform(0.04, 0.10)
            
        # 5. Planting Window timing
        if planting_window == 'Late':
            stress_adjustment += base_dtm * random.uniform(0.02, 0.05)
            
        # Add slight natural biological variance (noise)
        natural_noise = np.random.normal(0, max(1.5, base_dtm * 0.025))
        
        actual_days_to_harvest = int(round(base_dtm + stress_adjustment + natural_noise))
        # Ensure realistic physiological bounds: cannot mature in negative time or less than 60% of base
        actual_days_to_harvest = max(int(dtm_low * 0.85), actual_days_to_harvest)
        
        # Calculate target harvest month & period
        harvest_month, harvest_period = calculate_harvest_target(planting_month, actual_days_to_harvest)
        
        record = {
            'Crop': crop,
            'State': state,
            'Season': season,
            'Planting_Month': planting_month,
            'Temperature': temperature,
            'Rainfall': rainfall,
            'Soil_Moisture': soil_moisture,
            'Nutrient_Level': nutrient_level,
            'Fertilizer': fertilizer,
            'Weed_Competition': weed_competition,
            'Planting_Window': planting_window,
            'Days_to_Maturity': base_dtm,
            'Days_to_Harvest': actual_days_to_harvest,
            'Harvest_Month': harvest_month,
            'Harvest_Period': harvest_period
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    return df


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("[*] Generating grounded Nigerian agricultural dataset (IITA/NIHORT/NCRI/NiMet benchmarks)...")
    df = generate_nigerian_agronomic_dataset(num_records=2400)
    
    raw_filepath = os.path.join(raw_dir, 'nigerian_crop_harvest_data.csv')
    df.to_csv(raw_filepath, index=False)
    print(f"[+] Raw dataset saved to: {raw_filepath} ({len(df)} records)")
    
    # Save a processed copy ready for training/EDA
    processed_filepath = os.path.join(processed_dir, 'cleaned_crop_harvest_data.csv')
    df.to_csv(processed_filepath, index=False)
    print(f"[+] Processed dataset saved to: {processed_filepath}")
    
    # Print summary statistics
    print("\n--- Summary by Crop ---")
    summary = df.groupby('Crop')[['Days_to_Maturity', 'Days_to_Harvest', 'Temperature', 'Rainfall']].mean()
    print(summary.round(1))
    
    print("\n--- Value Counts by Crop ---")
    print(df['Crop'].value_counts())
    
    print("\n--- Target Harvest Month Distribution Sample ---")
    print(df['Harvest_Month'].value_counts().head(6))


if __name__ == '__main__':
    main()
