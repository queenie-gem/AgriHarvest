"""
AgriHarvest: A Machine Learning-Based Crop Harvest Time Prediction
and Farming Decision Support System
====================================================================
Main Streamlit Application — Fully compliant with AgriHarvest Master Spec.

Specification compliance:
- 5 supported crops only (§1)
- 10 supported states only (§2)
- 12 finalized input features only (§3)
- Primary target: Days_to_Harvest regression (§5)
- Approximate harvest window range without fabricated single dates (§1)
- Classification: secondary/experimental only (§6)
- Leakage disclosure: Days_to_Maturity correlation documented (§7)
- Out-of-scope features excluded from UI (§8)
- Input validation before prediction (§14)
- User-friendly output language (§15)
- Synthetic data labeled explicitly (§11)
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Add src to system path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predictor import HarvestPredictor

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriHarvest | Harvest Time Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        color: #1b5e20;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #f1f8e9;
        border-left: 6px solid #43a047;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .result-primary {
        font-size: 1.9rem;
        font-weight: 700;
        color: #2e7d32;
        margin: 0;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 14px;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 14px;
    }
    .disclaimer-box {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #555;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Resource Loading ──────────────────────────────────────────────────
@st.cache_resource
def get_predictor():
    models_dir = os.path.join(CURRENT_DIR, 'models')
    return HarvestPredictor(models_dir=models_dir)


@st.cache_data
def load_data():
    data_path  = os.path.join(CURRENT_DIR, 'data', 'processed', 'cleaned_crop_harvest_data.csv')
    reg_path   = os.path.join(CURRENT_DIR, 'models', 'regression_model_comparison.csv')
    clf_path   = os.path.join(CURRENT_DIR, 'models', 'classification_model_comparison.csv')

    df      = pd.read_csv(data_path) if os.path.exists(data_path) else None
    df_reg  = pd.read_csv(reg_path)  if os.path.exists(reg_path)  else None
    df_clf  = pd.read_csv(clf_path)  if os.path.exists(clf_path)  else None
    return df, df_reg, df_clf


# ─── Static Domain Data (matches spec §1, §2, and feature sets §3) ────────────
CROPS = ['Maize', 'Yam', 'Cassava', 'Tomato', 'Ewedu']
STATES = ['Oyo', 'Ogun', 'Ondo', 'Benue', 'Niger', 'Enugu', 'Kaduna', 'Plateau', 'Kano', 'Taraba']
MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December']
SEASONS = ['Rainy', 'Dry']
SOIL_MOISTURE_OPTS  = ['Low', 'Medium', 'High']
NUTRIENT_LEVEL_OPTS = ['Good', 'Fair', 'Poor']
FERTILIZER_OPTS     = ['NPK', 'Organic Manure', 'Urea', 'No Fertilizer']
WEED_OPTS           = ['Low', 'Medium', 'High']
PLANTING_WINDOW_OPTS= ['Optimal', 'Early', 'Late']

# Crop biological defaults — IITA / NIHORT / NCRI benchmarks
CROP_DEFAULTS = {
    'Maize':   {'dtm': 105, 'dtm_min': 80,  'dtm_max': 125, 'temp': 28.5, 'rain': 150.0},
    'Yam':     {'dtm': 230, 'dtm_min': 180, 'dtm_max': 270, 'temp': 28.0, 'rain': 110.0},
    'Cassava': {'dtm': 315, 'dtm_min': 270, 'dtm_max': 365, 'temp': 29.0, 'rain': 175.0},
    'Tomato':  {'dtm': 85,  'dtm_min': 70,  'dtm_max': 105, 'temp': 26.5, 'rain': 120.0},
    'Ewedu':   {'dtm': 38,  'dtm_min': 28,  'dtm_max': 50,  'temp': 28.5, 'rain': 160.0}
}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌾 AgriHarvest")
    st.markdown(
        "**A Machine Learning-Based Crop Harvest Time Prediction and "
        "Farming Decision Support System**\n\n"
        "Developed for the Nigerian agricultural context.\n\n"
        "**Supported crops (5):** Maize, Yam, Cassava, Tomato, Ewedu\n\n"
        "**Supported states (10):** Oyo, Ogun, Ondo, Benue, Niger, Enugu, "
        "Kaduna, Plateau, Kano, Taraba"
    )
    st.markdown("---")
    st.caption(
        "ℹ️ Dataset: 2,400 synthetically generated records grounded in "
        "agronomic benchmarks from IITA, NIHORT, NCRI, and NiMet. "
        "Not collected from actual field surveys."
    )


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌾 AgriHarvest</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A Machine Learning-Based Crop Harvest Time Prediction '
    '& Farming Decision Support System for Nigeria</div>',
    unsafe_allow_html=True
)
st.markdown("""
<div class="info-box">
<strong>📌 How it works:</strong> You select the crop you want to grow and describe your farming
conditions. AgriHarvest uses a trained machine learning model to estimate how many days until
your crop reaches maturity, and projects an approximate calendar harvest range.
The system does <strong>not</strong> decide which crop to grow — that is always your choice.
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Harvest Prediction",
    "📊 Data & Visualizations",
    "🔬 Model Evaluation",
    "ℹ️ About"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HARVEST PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Your Farming Conditions")
    st.write(
        "Fill in all 12 fields below. Each field corresponds to an official system input "
        "as defined in the AgriHarvest research specification."
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("##### Crop & Location")
        selected_crop = st.selectbox(
            "1. Crop *",
            options=CROPS,
            help="Select the crop you intend to cultivate. The model predicts harvest time — not the crop."
        )
        selected_state = st.selectbox(
            "2. Nigerian State *",
            options=STATES,
            help="State where cultivation will occur. Only the 10 listed states are supported."
        )
        selected_season = st.selectbox(
            "3. Farming Season *",
            options=SEASONS,
            help="Rainy (wet season) or Dry (irrigated / fadama production)."
        )
        selected_month = st.selectbox(
            "4. Planting Month *",
            options=MONTHS,
            index=4,
            help="Calendar month in which the crop will be sown or transplanted."
        )

    crop_d = CROP_DEFAULTS[selected_crop]

    with col2:
        st.markdown("##### Environmental Conditions")
        temperature = st.slider(
            "5. Average Temperature (°C) *",
            min_value=18.0, max_value=40.0,
            value=float(crop_d['temp']), step=0.5,
            help="Expected mean temperature during the crop's growing period."
        )
        rainfall = st.slider(
            "6. Expected Rainfall (mm) *",
            min_value=5.0, max_value=400.0,
            value=float(crop_d['rain']), step=5.0,
            help="Total rainfall expected over the growing season (mm)."
        )
        soil_moisture = st.selectbox(
            "7. Soil Moisture Level *",
            options=SOIL_MOISTURE_OPTS, index=1,
            help="Moisture availability in the crop root zone."
        )
        planting_window = st.selectbox(
            "11. Planting Window *",
            options=PLANTING_WINDOW_OPTS, index=0,
            help="Whether you are planting within the optimal agro-meteorological window for this crop."
        )

    with col3:
        st.markdown("##### Farm Management")
        nutrient_level = st.selectbox(
            "8. Soil Nutrient Level *",
            options=NUTRIENT_LEVEL_OPTS, index=0,
            help="Inherent or tested fertility status of the topsoil."
        )
        fertilizer = st.selectbox(
            "9. Fertilizer Applied *",
            options=FERTILIZER_OPTS, index=0,
            help="Primary fertilizer or nutrient source applied during the season."
        )
        weed_competition = st.selectbox(
            "10. Weed Competition *",
            options=WEED_OPTS, index=0,
            help="Degree of weed pressure during crop establishment and growth."
        )
        days_to_maturity = st.number_input(
            "12. Days to Maturity (cultivar baseline) *",
            min_value=crop_d['dtm_min'],
            max_value=crop_d['dtm_max'],
            value=crop_d['dtm'],
            step=1,
            help=(
                f"Standard biological maturity period for your chosen cultivar of {selected_crop}. "
                f"Typical range: {crop_d['dtm_min']}–{crop_d['dtm_max']} days. "
                "Check your seed packaging or consult an extension officer."
            )
        )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Harvest Period", type="primary", use_container_width=True)

    if predict_btn:
        # Input validation
        validation_errors = []
        if temperature < 18.0 or temperature > 40.0:
            validation_errors.append("Temperature must be between 18°C and 40°C.")
        if rainfall < 5.0 or rainfall > 400.0:
            validation_errors.append("Rainfall must be between 5 mm and 400 mm.")
        if days_to_maturity < crop_d['dtm_min'] or days_to_maturity > crop_d['dtm_max']:
            validation_errors.append(
                f"Days to Maturity for {selected_crop} must be between {crop_d['dtm_min']} and {crop_d['dtm_max']}."
            )

        if validation_errors:
            for err in validation_errors:
                st.error(f"⛔ {err}")
        else:
            predictor = get_predictor()
            input_data = {
                'Crop':             selected_crop,
                'State':            selected_state,
                'Season':           selected_season,
                'Planting_Month':   selected_month,
                'Temperature':      temperature,
                'Rainfall':         rainfall,
                'Soil_Moisture':    soil_moisture,
                'Nutrient_Level':   nutrient_level,
                'Fertilizer':       fertilizer,
                'Weed_Competition': weed_competition,
                'Planting_Window':  planting_window,
                'Days_to_Maturity': days_to_maturity
            }

            try:
                result = predictor.predict(input_data)

                # Primary result card — approximate calendar range without fabricated exact date
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:0.85rem;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
                        Estimated Harvest Window (Approximate Range)
                    </div>
                    <p class="result-primary">{result['expected_harvest_window']}</p>
                    <div style="margin-top:10px;font-size:1rem;color:#2e7d32;">
                        ⏱️ Estimated Cycle Duration: <strong>{result['predicted_days_to_harvest']} days</strong> (~{result['predicted_months_approx']} months)
                    </div>
                    <div style="margin-top:6px;font-size:0.88rem;color:#555;">
                        ℹ️ Window reflects sowing across <strong>{result['sowing_span']}</strong>. Actual harvest timing depends on your specific sowing day.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Primary metric row (conflicting Direct Month Estimate removed)
                m1, m2, m3 = st.columns(3)
                m1.metric("Estimated Cycle Duration", f"{result['predicted_days_to_harvest']} days", f"~{result['predicted_months_approx']} months")
                m2.metric("Cultivar Baseline (DTM)", f"{result['baseline_days_to_maturity']} days")
                delay = result['delay_days']
                m3.metric(
                    "Management/Stress Variance",
                    f"{delay:+d} days",
                    delta=f"{delay:+d} vs baseline",
                    delta_color="inverse"
                )

                # Decision support advisory
                st.subheader("🌱 Decision Support & Field Advisory")
                for note in result['advisory_notes']:
                    st.markdown(note)

                # Disclaimer
                st.markdown("""
                <div class="disclaimer-box">
                ⚠️ <strong>Disclaimer:</strong> AgriHarvest provides an estimated harvest window based on the
                supplied information and patterns learned from the available dataset. It does <strong>not</strong>
                guarantee an exact harvest date or farm yield. Combine this estimate with on-farm observation
                and consultation with an agricultural extension officer.
                </div>
                """, unsafe_allow_html=True)

            except ValueError as ve:
                st.error(f"⛔ Input Validation Error: {ve}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA & VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    df_data, _, _ = load_data()

    st.subheader("Agricultural Data Analysis")
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Data Transparency Notice:</strong> The AgriHarvest dataset (2,400 records) is
    <strong>synthetically generated</strong>. It was not collected through real field surveys.
    Its feature distributions and agronomic ranges are grounded in published benchmarks from
    IITA, NIHORT, NCRI, and NiMet — but the records themselves are simulated.
    </div>
    """, unsafe_allow_html=True)

    figures_dir = os.path.join(CURRENT_DIR, 'reports', 'figures')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Crop Distribution Across Nigerian States")
        p = os.path.join(figures_dir, 'fig1_crop_state_distribution.png')
        if os.path.exists(p): st.image(p, use_container_width=True)

        st.markdown("##### Stress Impact on Harvest Delay")
        p = os.path.join(figures_dir, 'fig3_stress_impact_delay.png')
        if os.path.exists(p): st.image(p, use_container_width=True)

    with c2:
        st.markdown("##### Days to Harvest Across the 5 Crops")
        p = os.path.join(figures_dir, 'fig2_crop_maturity_boxplot.png')
        if os.path.exists(p): st.image(p, use_container_width=True)

        st.markdown("##### Top Feature Importances")
        p = os.path.join(figures_dir, 'fig5_feature_importance.png')
        if os.path.exists(p): st.image(p, use_container_width=True)

    st.markdown("##### Climatic Pattern: Rainfall vs Temperature by Season")
    p = os.path.join(figures_dir, 'fig4_climate_scatter.png')
    if os.path.exists(p): st.image(p, use_container_width=True)

    with st.expander("🔍 Preview dataset sample (first 15 rows)"):
        if df_data is not None:
            st.dataframe(df_data.head(15), use_container_width=True)
        else:
            st.warning("Processed dataset file not found.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    _, df_reg, df_clf = load_data()

    st.subheader("Machine Learning Model Comparative Evaluation")
    st.write(
        "All candidate models were trained and evaluated on an independent 20% holdout test set "
        "(stratified by crop to ensure balanced representation). Results below are actual "
        "experimental outputs — not manually entered values."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Primary Model: Regression (Days_to_Harvest)")
        st.write("**Target: `Days_to_Harvest` (continuous numeric prediction)**")
        if df_reg is not None:
            st.dataframe(
                df_reg.style
                    .highlight_min(subset=['MAE (Days)', 'RMSE (Days)'], color='#c8e6c9')
                    .highlight_max(subset=['R2_Score'], color='#c8e6c9'),
                use_container_width=True
            )
        st.success(
            "🏆 **Selected Model:** **Gradient Boosting Regressor**  \n"
            "Achieved the lowest MAE (~5.04 days) and highest R² (~0.9950) on the test holdout."
        )

    with col_b:
        st.markdown("#### Secondary Experiment: Classification (Harvest_Month)")
        st.write(
            "**Target: `Harvest_Month` (12-class discrete prediction)**  \n"
            "*Experimental comparison only — not the primary prediction engine.*"
        )
        if df_clf is not None:
            st.dataframe(
                df_clf.style
                    .highlight_max(subset=['Accuracy', 'F1_Score'], color='#c8e6c9'),
                use_container_width=True
            )
        st.info(
            "ℹ️ **Experimental Observation:** Classification achieves ~71.9% accuracy because "
            "discrete monthly bins penalize boundary predictions harshly (e.g. predicting July 31 "
            "vs August 1 counts as a total miss). Regression estimates continuous duration, "
            "which allows calendar range projection without boundary discretization artifacts."
        )

    st.markdown("---")
    st.markdown("#### 🔬 Explaining Model Differences Near Month Boundaries")
    st.markdown("""
    When evaluating agricultural timelines, **continuous regression** and **discrete classification** can produce different outputs:
    
    1. **Continuous Regression (Primary Engine):** Predicts continuous days (e.g., 101 days). Sowing anytime in October projects a harvest window of *Late January – Early February*.
    2. **Discrete Classification (Secondary Experiment):** Predicts a single discrete categorical class (e.g., *February*). Because the classifier cannot represent multi-month intervals or early/mid/late periods, boundary cases fall into a single bucket.
    3. **Field Validation Notice:** Neither the regressor nor the classifier has been validated on real farm observations. Both models reflect patterns learned from synthetic training records.
    """)

    st.markdown("---")
    st.markdown("#### ⚠️ Research Integrity & Feature Evaluation Analysis")
    st.markdown("""
    As required by the research audit, feature relationships and target sensitivity were systematically evaluated:

    | Evaluation Condition | Test MAE | Test R² | Description |
    |---|---|---|---|
    | **With `Days_to_Maturity` (Approved Schema)** | **5.04 days** | **0.9950** | Standard 12-feature model where cultivar baseline is provided by user. |
    | **Without `Days_to_Maturity` (Ablation Test)** | **14.81 days** | **0.9614** | Leakage-free baseline predicting purely from weather and management. |

    **Key Findings:**
    - `Days_to_Maturity` and `Days_to_Harvest` exhibit a high Pearson correlation ($r = 0.9939$) because actual harvest time is naturally bounded by cultivar genetics.
    - **High correlation alone is not conclusive proof of target leakage**, because cultivar maturity is genuine external information available to the farmer (from seed packaging or extension guidelines).
    - However, because the synthetic target generator added only modest percentage adjustments (±2% to 15%) to baseline DTM, the baseline accounts for most of the target variance.
    - In accordance with research-scope change controls, `Days_to_Maturity` is retained as an approved input feature, with its dominant role documented transparently.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("About AgriHarvest")
    st.markdown("""
    **AgriHarvest** is an agricultural decision-support application built for Nigerian farming
    conditions. It helps prospective farmers, smallholders, and agricultural learners plan
    cultivation and anticipate harvest schedules with data-driven estimates.

    #### Core Design Principle
    > **The user selects the crop. The machine learning model predicts the expected harvest time.**

    AgriHarvest is not a crop recommendation system, yield predictor, disease detector, or
    replacement for professional agricultural expertise.

    ---
    #### The 5 Supported Crops
    | Crop | Institution Benchmark | Maturity Range |
    |---|---|---|
    | **Maize** (*Zea mays*) | NCRI / IITA | 90–120 days |
    | **Yam** (*Dioscorea spp.*) | IITA / NRCRI | 180–270 days |
    | **Cassava** (*Manihot esculenta*) | IITA | 270–365+ days |
    | **Tomato** (*Solanum lycopersicum*) | NIHORT | 75–100 days |
    | **Ewedu** (*Corchorus olitorius*) | NIHORT | 30–45 days |

    ---
    #### The 12 Official Input Features
    The system uses exactly these 12 features — no more, no less:
    `Crop`, `State`, `Season`, `Planting_Month`, `Temperature`, `Rainfall`,
    `Soil_Moisture`, `Nutrient_Level`, `Fertilizer`, `Weed_Competition`,
    `Planting_Window`, `Days_to_Maturity`.

    ---
    #### Agronomic & Benchmark Sources
    - **IITA** — International Institute of Tropical Agriculture, Ibadan
    - **NIHORT** — National Horticultural Research Institute, Ibadan
    - **NCRI** — National Cereals Research Institute, Badeggi
    - **NiMet** — Nigerian Meteorological Agency Seasonal Climate Predictions
    - **NBS / FAO** — National Bureau of Statistics and FAO crop production data benchmarks

    ---
    #### Supported States (10 Representative States)
    Oyo, Ogun, Ondo, Benue, Niger, Enugu, Kaduna, Plateau, Kano, Taraba.

    *The model is trained on these 10 representative states across Nigeria's major agro-ecological zones and has not been validated across all 36 Nigerian states.*

    ---
    #### Research Integrity & Dataset Notice
    1. **2,400 Synthetic Records:** The dataset was constructed from published agronomic literature and NiMet climatic distributions. It does not consist of real field surveys.
    2. **Decision Support Only:** Output ranges are machine-learning estimates, not guarantees.
    3. **Field Scouting Required:** Farmers should combine estimates with real field observations and consultation with agricultural extension agents.
    """)
