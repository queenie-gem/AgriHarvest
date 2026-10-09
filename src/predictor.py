"""
AgriHarvest: Harvest Prediction & Agronomic Advisory Engine
============================================================
Provides validated inference for the Streamlit web application.
Accepts user farming parameters, validates against approved schema,
executes the trained ML models, calculates approximate calendar
harvest ranges without fabricating exact planting dates, and provides
tailored agronomic decision support.
"""

import os
import calendar
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Approved Domain Constants & Feature Schema
VALID_CROPS = ['Maize', 'Yam', 'Cassava', 'Tomato', 'Ewedu']
VALID_STATES = ['Oyo', 'Ogun', 'Ondo', 'Benue', 'Niger', 'Enugu', 'Kaduna', 'Plateau', 'Kano', 'Taraba']
VALID_SEASONS = ['Rainy', 'Dry']
VALID_MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
VALID_SOIL_MOISTURE = ['Low', 'Medium', 'High']
VALID_NUTRIENT_LEVELS = ['Good', 'Fair', 'Poor']
VALID_FERTILIZERS = ['NPK', 'Organic Manure', 'Urea', 'No Fertilizer']
VALID_WEED_COMPETITION = ['Low', 'Medium', 'High']
VALID_PLANTING_WINDOWS = ['Optimal', 'Early', 'Late']

REQUIRED_FEATURES = [
    'Crop', 'State', 'Season', 'Planting_Month',
    'Temperature', 'Rainfall', 'Soil_Moisture', 'Nutrient_Level',
    'Fertilizer', 'Weed_Competition', 'Planting_Window', 'Days_to_Maturity'
]

NUMERIC_BOUNDS = {
    'Temperature': (15.0, 45.0),
    'Rainfall': (0.0, 500.0),
    'Days_to_Maturity': (20, 400)
}

MONTH_TO_INT = {m: i + 1 for i, m in enumerate(VALID_MONTHS)}
INT_TO_MONTH = {i + 1: m for i, m in enumerate(VALID_MONTHS)}


def get_period_prefix(day):
    """Returns Early, Mid, or Late sub-month period."""
    if day <= 10:
        return "Early"
    elif day <= 20:
        return "Mid"
    else:
        return "Late"


def calculate_harvest_window(planting_month_name, days_to_harvest, reference_year=2024):
    """
    Calculates an approximate calendar harvest range by considering planting
    dates spanning from the first through the last day of the selected month.
    
    Handles month boundaries, December-to-January transitions, and leap years
    without fabricating a specific single planting day.
    """
    month_int = MONTH_TO_INT[planting_month_name]
    num_days = calendar.monthrange(reference_year, month_int)[1]
    
    earliest_plant = datetime(reference_year, month_int, 1)
    latest_plant = datetime(reference_year, month_int, num_days)
    
    dth_delta = timedelta(days=int(round(days_to_harvest)))
    earliest_harvest = earliest_plant + dth_delta
    latest_harvest = latest_plant + dth_delta
    
    start_prefix = get_period_prefix(earliest_harvest.day)
    start_month = INT_TO_MONTH[earliest_harvest.month]
    start_label = f"{start_prefix} {start_month}"
    
    end_prefix = get_period_prefix(latest_harvest.day)
    end_month = INT_TO_MONTH[latest_harvest.month]
    end_label = f"{end_prefix} {end_month}"
    
    if start_label == end_label:
        harvest_window_str = start_label
    else:
        harvest_window_str = f"{start_label} - {end_label}"
        
    # Year transition documentation
    if earliest_harvest.year == latest_harvest.year:
        if earliest_harvest.year != reference_year:
            year_note = f"Harvest spans into the subsequent calendar year relative to sowing."
        else:
            year_note = "Harvest falls within the same calendar year as sowing."
    else:
        year_note = f"Harvest spans across the year transition boundary."
        
    return {
        'harvest_window': harvest_window_str,
        'earliest_harvest_period': start_label,
        'latest_harvest_period': end_label,
        'earliest_harvest_month': start_month,
        'latest_harvest_month': end_month,
        'sowing_span': f"1st – {num_days}th {planting_month_name}",
        'cycle_year_note': year_note
    }


def validate_input(input_dict):
    """
    Strict validation of prediction inputs against the approved schema.
    Rejects missing keys, invalid categories, non-numeric or non-finite values,
    and out-of-range figures with clear, descriptive ValueError exceptions.
    """
    if not isinstance(input_dict, dict):
        raise ValueError(f"Input must be a dictionary, got {type(input_dict).__name__}.")
        
    # Check for missing keys
    missing_keys = [k for k in REQUIRED_FEATURES if k not in input_dict]
    if missing_keys:
        raise ValueError(f"Missing required input features: {missing_keys}")
        
    # Check categorical features
    cat_checks = {
        'Crop': VALID_CROPS,
        'State': VALID_STATES,
        'Season': VALID_SEASONS,
        'Planting_Month': VALID_MONTHS,
        'Soil_Moisture': VALID_SOIL_MOISTURE,
        'Nutrient_Level': VALID_NUTRIENT_LEVELS,
        'Fertilizer': VALID_FERTILIZERS,
        'Weed_Competition': VALID_WEED_COMPETITION,
        'Planting_Window': VALID_PLANTING_WINDOWS
    }
    for feat, allowed in cat_checks.items():
        val = input_dict[feat]
        if val not in allowed:
            raise ValueError(f"Invalid categorical value for '{feat}': {val!r}. Must be one of: {allowed}")
            
    # Check numeric features
    for feat, (low, high) in NUMERIC_BOUNDS.items():
        val = input_dict[feat]
        if val is None or not isinstance(val, (int, float, np.number)):
            raise ValueError(f"Feature '{feat}' must be numeric, got {type(val).__name__} ({val!r}).")
        if not np.isfinite(val):
            raise ValueError(f"Feature '{feat}' must be finite, got non-finite value: {val}.")
        if not (low <= val <= high):
            raise ValueError(f"Feature '{feat}' out of valid range [{low}, {high}]: got {val}.")


class HarvestPredictor:
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            models_dir = os.path.join(base_dir, 'models')
            
        self.models_dir = models_dir
        self.reg_pipeline = None
        self.clf_pipeline = None
        self.metadata = None
        self._load_models()
        
    def _load_models(self):
        reg_path = os.path.join(self.models_dir, 'harvest_regressor.pkl')
        clf_path = os.path.join(self.models_dir, 'harvest_classifier.pkl')
        meta_path = os.path.join(self.models_dir, 'pipeline_metadata.pkl')
        
        if os.path.exists(reg_path):
            self.reg_pipeline = joblib.load(reg_path)
        if os.path.exists(clf_path):
            self.clf_pipeline = joblib.load(clf_path)
        if os.path.exists(meta_path):
            self.metadata = joblib.load(meta_path)
            
    def predict(self, input_dict):
        """
        Validates input parameters and processes prediction via trained pipelines.
        
        Returns:
            dict containing:
            - predicted_days_to_harvest (int)
            - predicted_months_approx (float)
            - baseline_days_to_maturity (int)
            - delay_days (int)
            - expected_harvest_window (str)
            - earliest_harvest_period (str)
            - latest_harvest_period (str)
            - earliest_harvest_month (str)
            - latest_harvest_month (str)
            - sowing_span (str)
            - cycle_year_note (str)
            - classifier_predicted_month (str)
            - advisory_notes (list of str)
        """
        # 1. Strict Validation Layer
        validate_input(input_dict)
        
        # Prepare DataFrame conforming to trained schema
        df_input = pd.DataFrame([input_dict])
        
        # 2. Regression Prediction (Primary Model: Continuous Days to Harvest)
        predicted_dth = self.reg_pipeline.predict(df_input)[0]
        predicted_dth = max(20.0, round(float(predicted_dth), 1))
        dth_int = int(round(predicted_dth))
        
        # 3. Secondary Classification Prediction (Experimental Comparison)
        clf_month = self.clf_pipeline.predict(df_input)[0] if self.clf_pipeline is not None else "N/A"
        
        # 4. Approximate Harvest Window Calculation (No Fabricated Single Day)
        plant_month = input_dict['Planting_Month']
        window_info = calculate_harvest_window(plant_month, dth_int)
        
        # 5. Maturity Variance Analysis
        base_dtm = int(round(input_dict['Days_to_Maturity']))
        delay_days = int(round(dth_int - base_dtm))
        
        # 6. Agricultural Advisory Generation
        advisory_notes = self._generate_advisory(input_dict, delay_days)
        
        return {
            'crop': input_dict['Crop'],
            'state': input_dict['State'],
            'planting_month': plant_month,
            'predicted_days_to_harvest': dth_int,
            'predicted_months_approx': round(dth_int / 30.4, 1),
            'baseline_days_to_maturity': base_dtm,
            'delay_days': delay_days,
            'expected_harvest_window': window_info['harvest_window'],
            'expected_harvest_period': window_info['harvest_window'],  # Backwards compatibility
            'earliest_harvest_period': window_info['earliest_harvest_period'],
            'latest_harvest_period': window_info['latest_harvest_period'],
            'earliest_harvest_month': window_info['earliest_harvest_month'],
            'latest_harvest_month': window_info['latest_harvest_month'],
            'sowing_span': window_info['sowing_span'],
            'cycle_year_note': window_info['cycle_year_note'],
            'classifier_predicted_month': clf_month,
            'advisory_notes': advisory_notes
        }
        
    def _generate_advisory(self, inputs, delay_days):
        notes = []
        crop = inputs['Crop']
        
        # Delay / Stress commentary
        if delay_days > 7:
            notes.append(
                f"⚠️ Field conditions are projected to extend maturity by approximately **{delay_days} days** "
                "beyond the crop's baseline timeline due to environmental or management stress."
            )
        elif delay_days < -3:
            notes.append(
                f"⚡ Optimal growing conditions are projected to accelerate physiological maturity by "
                f"approximately **{abs(delay_days)} days** relative to the cultivar baseline."
            )
        else:
            notes.append("✅ Crop growth is tracking closely to standard physiological maturity benchmarks.")
            
        # Weed Competition
        if inputs['Weed_Competition'] == 'High':
            notes.append(
                "🌿 **High Weed Competition Detected:** Weeds compete fiercely for sunlight, soil moisture, and nitrogen. "
                "Timely weeding or mulching is advised within the first 3-6 weeks to prevent severe yield loss."
            )
        elif inputs['Weed_Competition'] == 'Medium':
            notes.append("🌿 **Moderate Weed Pressure:** Schedule a secondary weeding operation prior to canopy closure.")
            
        # Soil Moisture & Season
        if inputs['Season'] == 'Dry' and inputs['Soil_Moisture'] == 'Low':
            notes.append(
                "💧 **Water Deficit Alert:** Planting in the dry season with low soil moisture requires supplementary "
                "fadama / drip irrigation to avoid crop failure or stunted bulking."
            )
            
        # Nutrient Level & Fertilizer
        if inputs['Nutrient_Level'] == 'Poor' and inputs['Fertilizer'] == 'No Fertilizer':
            notes.append(
                "🧪 **Nutrient Deficiency Alert:** Soil nutrient status is poor with zero fertilizer applied. "
                "Applying balanced NPK or well-decomposed organic manure will significantly improve crop development."
            )
        elif inputs['Fertilizer'] == 'Organic Manure':
            notes.append("🌱 **Soil Health:** Organic manure improves soil microbial life and moisture retention.")
            
        # Crop-specific tips
        if crop == 'Yam' and inputs['Planting_Window'] == 'Early':
            notes.append("🍠 **Yam Production Tip:** Traditional early mound planting allows sets to break dormancy as the rainy season initiates.")
        elif crop == 'Cassava':
            notes.append("🌿 **Cassava Management:** Ensure clean stem cuttings (20-25cm) placed at an angle with at least 4-6 nodes buried.")
        elif crop == 'Tomato' and inputs['Temperature'] > 32:
            notes.append("🍅 **Tomato Heat Advisory:** High daytime temperatures above 32°C can cause blossom drop and fruit scald. Ensure shade or adequate soil moisture.")
        elif crop == 'Ewedu':
            notes.append("🥬 **Ewedu Harvesting:** Harvest by cutting shoots 5-10cm above ground to encourage ratoon regrowth for subsequent flushes.")
            
        return notes
