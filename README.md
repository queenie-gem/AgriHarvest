# AgriHarvest: A Machine Learning-Based Crop Harvest Time Prediction and Farming Decision Support System

## 1. Project Overview
**AgriHarvest** is an agricultural decision-support application specifically designed for the **Nigerian agricultural context**. 

In conventional agricultural applications, models frequently attempt to classify which crop a farmer should grow. In **AgriHarvest**, a fundamental design principle is observed:

> **The user selects the crop. The machine learning model does NOT predict the crop.**
> Instead, the model analyzes the user's field conditions, location, and management practices to predict the **expected harvest period**.

The system provides data-driven guidance to prospective farmers, smallholders, agricultural learners, and existing farmers across Nigeria's major agro-ecological zones.

---

## 2. The Five Covered Crops
The system strictly supports the 5 focal crops established in the research scope:
1. **Maize (*Zea mays*)** — Maturity: ~90–120 days. Major staple cereal grown nationwide.
2. **Yam (*Dioscorea spp.*)** — Maturity: ~180–270 days (6–9 months). Premier root crop in the Nigerian Middle Belt & South.
3. **Cassava (*Manihot esculenta*)** — Maturity: ~270–365+ days (9–12+ months). Nigeria's key food security staple.
4. **Tomato (*Solanum lycopersicum*)** — Maturity: ~75–100 days. High-value horticultural crop with intensive dry-season production.
5. **Ewedu (*Corchorus olitorius* / Jute Mallow)** — Maturity: ~30–45 days. Fast-turnover indigenous leafy vegetable.

---

## 3. The 12 Finalized Input Features
The system uses strictly the 12 finalized features (no unauthorized features such as soil pH or water requirement):

| # | Feature | Type | Description / Units |
|---|---|---|---|
| 1 | **Crop** | Categorical | User-selected crop (Maize, Yam, Cassava, Tomato, Ewedu) |
| 2 | **State** | Categorical | Nigerian state (Oyo, Ogun, Ondo, Benue, Niger, Enugu, Kaduna, Plateau, Kano, Taraba) |
| 3 | **Season** | Categorical | Farming season (Rainy, Dry) |
| 4 | **Planting_Month** | Categorical | Month of planting (January – December) |
| 5 | **Temperature** | Numeric | Average environmental temperature in °C |
| 6 | **Rainfall** | Numeric | Total rainfall in mm |
| 7 | **Soil_Moisture** | Categorical | Soil moisture availability (Low, Medium, High) |
| 8 | **Nutrient_Level** | Categorical | Topsoil fertility condition (Poor, Fair, Good) |
| 9 | **Fertilizer** | Categorical | Fertilizer applied (NPK, Urea, Organic Manure, No Fertilizer) |
| 10 | **Weed_Competition** | Categorical | Weed pressure level (Low, Medium, High) |
| 11 | **Planting_Window** | Categorical | Window timing (Early, Optimal, Late) |
| 12 | **Days_to_Maturity** | Numeric | Standard biological maturity duration for cultivar in days |

**Primary Prediction Target:**
- **`Days_to_Harvest`:** Predicted continuous days to maturity under field conditions, converted to an approximate **calendar harvest range** (e.g. *Late January – Early February*).

---

## 4. Scientific Agronomic Grounding & Synthetic Data Disclosure
The dataset contains **2,400 synthetically generated records** whose distributions and agronomic ranges are grounded in published benchmarks from:
- **IITA** — International Institute of Tropical Agriculture, Ibadan, Nigeria
- **NIHORT** — National Horticultural Research Institute, Ibadan, Nigeria
- **NCRI** — National Cereals Research Institute, Badeggi, Niger State
- **NiMet** — Nigerian Meteorological Agency Seasonal Climate Predictions (SCP)
- **NBS / FAO** — National Bureau of Statistics and FAO crop production benchmarks

*Notice: The records are simulated from agronomic literature benchmarks and do not constitute actual field surveys or farmer interviews.*

---

## 5. Machine Learning Benchmarks & Model Selection
Following empirical evaluation on an independent 20% holdout test set (stratified by crop):

### A. Primary Regression Models (Target: `Days_to_Harvest`)
| Model | MAE (Days) | RMSE (Days) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting Regressor** *(Selected)* | **5.04** | **7.64** | **0.9950** |
| Random Forest Regressor | 5.99 | 9.13 | 0.9928 |
| Linear Regression | 6.43 | 9.25 | 0.9926 |
| Ridge Regression | 6.44 | 9.28 | 0.9926 |
| Decision Tree Regressor | 7.58 | 11.17 | 0.9892 |

### B. Secondary Classification Experiment (Target: `Harvest_Month`)
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier** | **71.88%** | **0.7278** | **0.7188** | **0.7192** |
| Random Forest Classifier | 69.79% | 0.7134 | 0.6979 | 0.6997 |
| Decision Tree Classifier | 58.75% | 0.6999 | 0.5875 | 0.5984 |
| Logistic Regression | 47.08% | 0.4622 | 0.4708 | 0.4604 |

### Feature Correlation & Ablation Analysis:
- In testing, removing `Days_to_Maturity` yields: MAE = 14.81 days, $R^2 = 0.9614$.
- `Days_to_Maturity` has an $r = 0.9939$ correlation with `Days_to_Harvest` because harvest duration is fundamentally bounded by cultivar genetics.
- High correlation alone is not conclusive proof of target leakage when cultivar maturity is genuinely supplied by the farmer, but does show that DTM is the dominant driver in the synthetic generator's output.

---

## 6. Directory Structure
```text
smart-season/
│
├── data/
│   ├── raw/
│   │   └── nigerian_crop_harvest_data.csv   # Grounded 2,400 benchmark records
│   └── processed/
│       └── cleaned_crop_harvest_data.csv    # Cleaned, validated dataset
│
├── models/
│   ├── harvest_regressor.pkl                # Trained Gradient Boosting Regressor pipeline
│   ├── harvest_classifier.pkl               # Trained Gradient Boosting Classifier pipeline
│   ├── pipeline_metadata.pkl                # Feature metadata and schema definition
│   ├── regression_model_comparison.csv      # Empirical regression metrics table
│   └── classification_model_comparison.csv  # Empirical classification metrics table
│
├── reports/
│   └── figures/
│       ├── fig1_crop_state_distribution.png # Distribution across Nigerian states
│       ├── fig2_crop_maturity_boxplot.png   # Maturity boxplot across the 5 crops
│       ├── fig3_stress_impact_delay.png     # Impact of weed & nutrient stress
│       ├── fig4_climate_scatter.png         # Rainfall & temperature patterns
│       └── fig5_feature_importance.png      # Top 10 Gradient Boosting feature importances
│
├── src/
│   ├── data_preparation.py                  # Agronomic data generator & cleaner
│   ├── preprocessing.py                     # Scikit-learn ColumnTransformer pipeline
│   ├── train.py                             # ML training, cross-validation & benchmark
│   ├── visualization.py                     # Academic figure generator
│   ├── predictor.py                         # Validated inference & range projection engine
│   ├── validate.py                          # 18-point compliance validation suite
│   └── test_suite.py                        # Comprehensive 7-category unit test suite
│
├── app.py                                   # Streamlit Interactive Web Application
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## 7. How to Run the Application

To launch the interactive Streamlit dashboard:

```powershell
# Navigate to the project directory
cd C:\Users\ASUS\.gemini\antigravity\scratch\smart-season

# Launch the Streamlit web application
python -m streamlit run app.py
```

The application will be accessible in your web browser at `http://localhost:8501`.

---

## 8. System Limitations & Ethical Principles
1. **Decision Support Tool:** AgriHarvest provides machine learning estimates, not yield or harvest date guarantees.
2. **Biological & Environmental Variability:** Extreme anomalies such as flash flooding, localized pest infestations, or unseasonal Harmattan dry spells cannot be captured in a statistical model.
3. **Local Microclimates & Extension Advice:** Predictions should always be combined with physical farm scouting and guidance from professional agricultural extension officers.
