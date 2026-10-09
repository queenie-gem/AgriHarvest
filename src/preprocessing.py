"""
Smart Season: Preprocessing and Feature Engineering Pipeline
============================================================
Prepares raw agricultural data for model training and provides reusable
pipelines for inference in the Streamlit dashboard.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CATEGORICAL_FEATURES = [
    'Crop', 'State', 'Season', 'Planting_Month',
    'Soil_Moisture', 'Nutrient_Level', 'Fertilizer',
    'Weed_Competition', 'Planting_Window'
]

NUMERICAL_FEATURES = [
    'Temperature', 'Rainfall', 'Days_to_Maturity'
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def get_preprocessor():
    """
    Constructs a ColumnTransformer that one-hot encodes categorical variables
    and standardizes numerical variables.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def prepare_datasets(data_path=None, test_size=0.2, random_state=42):
    """
    Loads processed dataset, splits into train/test sets for both
    Regression (Days_to_Harvest) and Classification (Harvest_Month).
    """
    if data_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_crop_harvest_data.csv')
        
    df = pd.read_csv(data_path)
    
    X = df[FEATURE_COLUMNS]
    y_reg = df['Days_to_Harvest']
    y_clf = df['Harvest_Month']
    
    # Train-test split (stratified by Crop to ensure balanced representation)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state, stratify=df['Crop']
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_reg_train': y_reg_train,
        'y_reg_test': y_reg_test,
        'y_clf_train': y_clf_train,
        'y_clf_test': y_clf_test,
        'feature_names': FEATURE_COLUMNS,
        'categorical_features': CATEGORICAL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES
    }


if __name__ == '__main__':
    data_dict = prepare_datasets()
    print(f"Data split successfully: {len(data_dict['X_train'])} train, {len(data_dict['X_test'])} test.")
