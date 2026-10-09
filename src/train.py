"""
Smart Season: Machine Learning Model Training & Comparative Evaluation
======================================================================
Trains and benchmarks multiple candidate models for both:
1. Regression (predicting Days_to_Harvest)
2. Classification (predicting Harvest_Month)

Evaluates models using standard objective metrics:
- Regression: MAE, MSE, RMSE, R2
- Classification: Accuracy, Precision, Recall, F1-Score

Saves best pipeline models to `models/` directory for Streamlit inference.
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

from preprocessing import get_preprocessor, prepare_datasets


def evaluate_regression_models(data_dict):
    """
    Trains and benchmarks regression models predicting Days_to_Harvest.
    """
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_reg_train']
    y_test = data_dict['y_reg_test']
    
    candidate_regressors = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    
    print("\n" + "="*60)
    print("      BENCHMARKING REGRESSION MODELS (Target: Days_to_Harvest)")
    print("="*60)
    
    for name, model in candidate_regressors.items():
        preprocessor = get_preprocessor()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Problem_Type': 'Regression',
            'MAE (Days)': round(mae, 2),
            'RMSE (Days)': round(rmse, 2),
            'R2_Score': round(r2, 4)
        })
        
        trained_pipelines[name] = (pipeline, r2)
        print(f"[{name}] -> MAE: {mae:.2f} days | RMSE: {rmse:.2f} days | R²: {r2:.4f}")
        
    df_reg_results = pd.DataFrame(results).sort_values(by='R2_Score', ascending=False)
    
    # Pick the best model based on R2
    best_reg_name = df_reg_results.iloc[0]['Model']
    best_reg_pipeline = trained_pipelines[best_reg_name][0]
    print(f"\n[*] Best Regression Model Selected: {best_reg_name} (R² = {df_reg_results.iloc[0]['R2_Score']})")
    
    return df_reg_results, best_reg_name, best_reg_pipeline


def evaluate_classification_models(data_dict):
    """
    Trains and benchmarks classification models predicting Harvest_Month.
    """
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_clf_train']
    y_test = data_dict['y_clf_test']
    
    candidate_classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree Classifier': DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    
    print("\n" + "="*60)
    print("    BENCHMARKING CLASSIFICATION MODELS (Target: Harvest_Month)")
    print("="*60)
    
    for name, model in candidate_classifiers.items():
        preprocessor = get_preprocessor()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Problem_Type': 'Classification',
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4)
        })
        
        trained_pipelines[name] = (pipeline, f1)
        print(f"[{name}] -> Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")
        
    df_clf_results = pd.DataFrame(results).sort_values(by='F1_Score', ascending=False)
    
    # Pick best classifier based on F1
    best_clf_name = df_clf_results.iloc[0]['Model']
    best_clf_pipeline = trained_pipelines[best_clf_name][0]
    print(f"\n[*] Best Classification Model Selected: {best_clf_name} (F1 = {df_clf_results.iloc[0]['F1_Score']})")
    
    return df_clf_results, best_clf_name, best_clf_pipeline


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("[*] Loading and splitting dataset...")
    data_dict = prepare_datasets()
    
    # 1. Run Regression Benchmarking
    df_reg, best_reg_name, best_reg_pipeline = evaluate_regression_models(data_dict)
    
    # 2. Run Classification Benchmarking
    df_clf, best_clf_name, best_clf_pipeline = evaluate_classification_models(data_dict)
    
    # Save comparison metrics table for the dashboard
    reg_csv_path = os.path.join(models_dir, 'regression_model_comparison.csv')
    clf_csv_path = os.path.join(models_dir, 'classification_model_comparison.csv')
    df_reg.to_csv(reg_csv_path, index=False)
    df_clf.to_csv(clf_csv_path, index=False)
    print(f"\n[+] Saved model evaluation metrics to: {models_dir}")
    
    # Save best models
    reg_model_path = os.path.join(models_dir, 'harvest_regressor.pkl')
    clf_model_path = os.path.join(models_dir, 'harvest_classifier.pkl')
    joblib.dump(best_reg_pipeline, reg_model_path)
    joblib.dump(best_clf_pipeline, clf_model_path)
    print(f"[+] Saved Best Regressor to: {reg_model_path}")
    print(f"[+] Saved Best Classifier to: {clf_model_path}")
    
    # Save metadata dictionary for fast inference and UI validation
    metadata = {
        'best_regressor_name': best_reg_name,
        'best_classifier_name': best_clf_name,
        'feature_columns': data_dict['feature_names'],
        'categorical_features': data_dict['categorical_features'],
        'numerical_features': data_dict['numerical_features']
    }
    joblib.dump(metadata, os.path.join(models_dir, 'pipeline_metadata.pkl'))
    print("[+] Saved Pipeline Metadata.")


if __name__ == '__main__':
    main()
