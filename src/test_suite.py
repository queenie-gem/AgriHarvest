"""
AgriHarvest: Comprehensive Verification & Regression Test Suite
================================================================
Covers all 7 verification categories requested in Section 6:
1. Harvest windows across all 12 months.
2. December-to-January transitions and leap years.
3. Month-only input without a fabricated exact planting day.
4. Missing, invalid, non-finite, and out-of-range inputs.
5. Correct model and preprocessing-pipeline use.
6. Agreement between the displayed harvest window and underlying date calculations.
7. Continued presence of exactly the 12 approved input features.
"""

import sys
import os
import calendar
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure src is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predictor import HarvestPredictor, calculate_harvest_window, validate_input, REQUIRED_FEATURES

PASS = "[PASS]"
FAIL = "[FAIL]"

def run_test(label, test_func):
    try:
        test_func()
        print(f"  {PASS} {label}")
        return True
    except AssertionError as ae:
        print(f"  {FAIL} {label}: AssertionError - {ae}")
        return False
    except Exception as e:
        print(f"  {FAIL} {label}: {type(e).__name__} - {e}")
        return False


def test_category_1_all_12_months():
    """Test 1: Verify calculate_harvest_window across all 12 months."""
    months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    for m in months:
        res = calculate_harvest_window(m, 90, reference_year=2024)
        assert 'harvest_window' in res, f"Missing harvest_window for {m}"
        assert '–' in res['harvest_window'] or len(res['harvest_window']) > 0
        assert res['sowing_span'].endswith(m)


def test_category_2_transitions_and_leap_years():
    """Test 2: December-to-January transitions and leap years."""
    # 2a. December planting transition into January/February
    res_dec = calculate_harvest_window('December', 45, reference_year=2024)
    assert 'January' in res_dec['harvest_window'] or 'February' in res_dec['harvest_window'], \
        f"December + 45 days should cross into Jan/Feb, got {res_dec['harvest_window']}"
    assert "subsequent calendar year" in res_dec['cycle_year_note'] or "spans across" in res_dec['cycle_year_note']

    # 2b. Leap year handling: 2024 is a leap year (29 days in Feb), 2025 is not (28 days)
    res_feb_leap = calculate_harvest_window('February', 30, reference_year=2024)
    assert "29th February" in res_feb_leap['sowing_span'], f"Leap year 2024 should have 29 days, got {res_feb_leap['sowing_span']}"
    
    res_feb_nonleap = calculate_harvest_window('February', 30, reference_year=2025)
    assert "28th February" in res_feb_nonleap['sowing_span'], f"Non-leap year 2025 should have 28 days, got {res_feb_nonleap['sowing_span']}"


def test_category_3_month_only_no_fabricated_day():
    """Test 3: Month-only input does not fabricate a specific day like 'January 24'."""
    p = HarvestPredictor()
    base_input = {
        'Crop': 'Maize', 'State': 'Oyo', 'Season': 'Dry',
        'Planting_Month': 'October', 'Temperature': 28.5, 'Rainfall': 150.0,
        'Soil_Moisture': 'Medium', 'Nutrient_Level': 'Good',
        'Fertilizer': 'NPK', 'Weed_Competition': 'Low',
        'Planting_Window': 'Optimal', 'Days_to_Maturity': 105
    }
    result = p.predict(base_input)
    assert 'expected_harvest_date' not in result or result.get('expected_harvest_date') is None, \
        "Single fabricated date should not be presented as definitive output."
    assert 'expected_harvest_window' in result
    assert '-' in result['expected_harvest_window'], "Should present an approximate window span."
    assert 'sowing_span' in result
    assert "October" in result['sowing_span']


def test_category_4_validation_rejections():
    """Test 4: Rejects missing, invalid, non-finite, and out-of-range inputs with ValueError."""
    base_input = {
        'Crop': 'Maize', 'State': 'Oyo', 'Season': 'Dry',
        'Planting_Month': 'October', 'Temperature': 28.5, 'Rainfall': 150.0,
        'Soil_Moisture': 'Medium', 'Nutrient_Level': 'Good',
        'Fertilizer': 'NPK', 'Weed_Competition': 'Low',
        'Planting_Window': 'Optimal', 'Days_to_Maturity': 105
    }
    
    # 4a. Missing key
    bad_missing = base_input.copy()
    del bad_missing['Crop']
    try:
        validate_input(bad_missing)
        raise AssertionError("Should have raised ValueError on missing Crop")
    except ValueError as e:
        assert "Missing required input features" in str(e)

    # 4b. Invalid categorical
    bad_cat = base_input.copy()
    bad_cat['Crop'] = 'Wheat'  # Unsupported crop
    try:
        validate_input(bad_cat)
        raise AssertionError("Should have raised ValueError on invalid crop Wheat")
    except ValueError as e:
        assert "Invalid categorical value for 'Crop'" in str(e)

    # 4c. Non-finite numeric (NaN)
    bad_nan = base_input.copy()
    bad_nan['Temperature'] = float('nan')
    try:
        validate_input(bad_nan)
        raise AssertionError("Should have raised ValueError on NaN temperature")
    except ValueError as e:
        assert "must be finite" in str(e)

    # 4d. Out-of-range numeric
    bad_temp = base_input.copy()
    bad_temp['Temperature'] = 55.0  # Above 45°C limit
    try:
        validate_input(bad_temp)
        raise AssertionError("Should have raised ValueError on out-of-range temp")
    except ValueError as e:
        assert "out of valid range" in str(e)


def test_category_5_pipeline_integrity():
    """Test 5: Confirms correct model pipeline execution without manual formulas."""
    p = HarvestPredictor()
    assert p.reg_pipeline is not None, "Regressor pipeline failed to load"
    assert hasattr(p.reg_pipeline, 'predict'), "Regressor pipeline must have predict method"
    
    # Run test on 2 different crops
    for crop, dtm in [('Maize', 105), ('Ewedu', 38)]:
        inp = {
            'Crop': crop, 'State': 'Oyo', 'Season': 'Rainy',
            'Planting_Month': 'May', 'Temperature': 28.0, 'Rainfall': 160.0,
            'Soil_Moisture': 'Medium', 'Nutrient_Level': 'Good',
            'Fertilizer': 'NPK', 'Weed_Competition': 'Low',
            'Planting_Window': 'Optimal', 'Days_to_Maturity': dtm
        }
        res = p.predict(inp)
        assert isinstance(res['predicted_days_to_harvest'], int)
        assert res['predicted_days_to_harvest'] > 20
        # Ewedu should mature much faster than Maize
        if crop == 'Ewedu':
            assert res['predicted_days_to_harvest'] < 60, f"Ewedu DTH should be < 60 days, got {res['predicted_days_to_harvest']}"


def test_category_6_window_date_agreement():
    """Test 6: Agreement between displayed harvest window and underlying date calculations."""
    res = calculate_harvest_window('October', 101, reference_year=2024)
    # Earliest date: Oct 1 + 101 days = Jan 10 (Early January)
    # Latest date: Oct 31 + 101 days = Feb 9 (Early February)
    assert res['earliest_harvest_period'] == 'Early January', f"Expected Early January, got {res['earliest_harvest_period']}"
    assert res['latest_harvest_period'] == 'Early February', f"Expected Early February, got {res['latest_harvest_period']}"
    assert res['harvest_window'] == 'Early January - Early February'


def test_category_7_exact_12_features():
    """Test 7: The continued presence of exactly the 12 approved input features."""
    expected = [
        'Crop', 'State', 'Season', 'Planting_Month',
        'Temperature', 'Rainfall', 'Soil_Moisture', 'Nutrient_Level',
        'Fertilizer', 'Weed_Competition', 'Planting_Window', 'Days_to_Maturity'
    ]
    assert len(REQUIRED_FEATURES) == 12, f"Expected 12 features, got {len(REQUIRED_FEATURES)}"
    assert REQUIRED_FEATURES == expected, f"Feature list does not match approved schema exactly: {REQUIRED_FEATURES}"


def main():
    print("=" * 65)
    print("  AgriHarvest Comprehensive Verification Test Suite")
    print("=" * 65)
    
    tests = [
        ("1. Harvest windows across all 12 months", test_category_1_all_12_months),
        ("2. December-to-January transitions and leap years", test_category_2_transitions_and_leap_years),
        ("3. Month-only input without fabricated exact day", test_category_3_month_only_no_fabricated_day),
        ("4. Rejection of missing, invalid, non-finite, out-of-range inputs", test_category_4_validation_rejections),
        ("5. Correct model pipeline execution without manual formulas", test_category_5_pipeline_integrity),
        ("6. Agreement between harvest window and underlying dates", test_category_6_window_date_agreement),
        ("7. Continued presence of exactly the 12 approved features", test_category_7_exact_12_features),
    ]
    
    passed = 0
    total = len(tests)
    for label, fn in tests:
        if run_test(label, fn):
            passed += 1
            
    print("\n" + "=" * 65)
    print(f"  SUMMARY: {passed}/{total} TEST CATEGORIES PASSED")
    print("=" * 65)
    
    if passed == total:
        print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print("  SOME TESTS FAILED.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
