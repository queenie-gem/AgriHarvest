"""
Smart Season: Agricultural Exploratory Data Analysis & Visualization
====================================================================
Generates comprehensive visual plots for understanding relationships
between farming management, climatic stress, and crop harvest timing in Nigeria.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for academic presentation and clarity
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def generate_all_visualizations():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_crop_harvest_data.csv')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(figures_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    print("[*] Generating Agricultural Visualizations...")
    
    # --- Figure 1: Crop Distribution across Nigerian Agro-Ecological States ---
    plt.figure(figsize=(10, 5))
    crop_state_ct = pd.crosstab(df['State'], df['Crop'])
    crop_state_ct.plot(kind='bar', stacked=True, colormap='tab10', figsize=(10, 5), edgecolor='black', alpha=0.85)
    plt.title('Crop Distribution Across Selected Nigerian States', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Nigerian State', fontsize=11)
    plt.ylabel('Number of Field Records', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Crop', frameon=True)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, 'fig1_crop_state_distribution.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {fig1_path}")
    
    # --- Figure 2: Days to Harvest vs Baseline Maturity Across 5 Crops ---
    plt.figure(figsize=(10, 5))
    df_sorted = df.sort_values(by='Days_to_Maturity')
    sns.boxplot(data=df_sorted, x='Crop', y='Days_to_Harvest', hue='Crop', palette='Set2', legend=False)
    plt.title('Days to Harvest Variation Across the 5 Selected Crops', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Crop', fontsize=11)
    plt.ylabel('Actual Days to Harvest (DTH)', fontsize=11)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, 'fig2_crop_maturity_boxplot.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {fig2_path}")
    
    # --- Figure 3: Stress Impact (Weed Competition & Nutrient Level) on Harvest Delay ---
    df['Harvest_Delay_Days'] = df['Days_to_Harvest'] - df['Days_to_Maturity']
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='Weed_Competition', y='Harvest_Delay_Days', hue='Nutrient_Level',
                palette='coolwarm', errorbar=None, edgecolor='black')
    plt.title('Impact of Weed Competition & Soil Nutrient Status on Harvest Delay', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Weed Competition Level', fontsize=11)
    plt.ylabel('Average Delay Beyond Baseline Maturity (Days)', fontsize=11)
    plt.legend(title='Nutrient Level', frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, 'fig3_stress_impact_delay.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {fig3_path}")
    
    # --- Figure 4: Climatic Patterns (Rainfall vs Temperature by Season) ---
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='Temperature', y='Rainfall', hue='Season', style='Season',
                    palette={'Rainy': '#2b83ba', 'Dry': '#d7191c'}, alpha=0.7, s=50)
    plt.title('Climatic Distribution: Growing Season Rainfall vs. Temperature in Nigeria', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Temperature (°C)', fontsize=11)
    plt.ylabel('Rainfall (mm)', fontsize=11)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, 'fig4_climate_scatter.png')
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {fig4_path}")
    
    # --- Figure 5: Feature Importance from Trained Gradient Boosting Regressor ---
    model_path = os.path.join(models_dir, 'harvest_regressor.pkl')
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        regressor = pipeline.named_steps['regressor']
        preprocessor = pipeline.named_steps['preprocessor']
        
        feature_names = []
        # Numerical features
        feature_names.extend(preprocessor.transformers_[0][2])
        # Categorical features one-hot encoded
        cat_encoder = preprocessor.transformers_[1][1]
        cat_feature_names = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names.extend(cat_feature_names)
        
        importances = regressor.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        top_feats = feat_df.sort_values(by='Importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(data=top_feats, x='Importance', y='Feature', hue='Feature', palette='viridis', edgecolor='black', legend=False)
        plt.title('Top 10 Feature Importances in Predicting Crop Harvest Period', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Relative Feature Importance (Gradient Boosting)', fontsize=11)
        plt.ylabel('Feature', fontsize=11)
        plt.tight_layout()
        fig5_path = os.path.join(figures_dir, 'fig5_feature_importance.png')
        plt.savefig(fig5_path, dpi=300)
        plt.close()
        print(f"[+] Saved: {fig5_path}")
        
    print("[*] All visualizations created successfully.")


if __name__ == '__main__':
    generate_all_visualizations()
