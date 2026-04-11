import streamlit as st
import joblib
import numpy as np
import pandas as pd # Import pandas for DataFrame creation
import re # Import re for amenity score function

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Nairobi Property Price Predictor",
    page_icon="🏡",
    layout="centered",
)

# ── Load model bundle (cached so it only loads once) ──────
@st.cache_resource
def load_model():
    """Load model.pkl — contains model, mae, and class lists."""
    return joblib.load("model.pkl")

bundle   = load_model()
model    = bundle["model"]
MAE      = bundle["mae"]
ALL_AMENITIES = bundle["ALL_AMENITIES"]
CAT_FEATURES = bundle["CAT_FEATURES"]
NUM_FEATURES = bundle["NUM_FEATURES"]
LOCATIONS_FOR_DROPDOWN = bundle["LOCATIONS_FOR_DROPDOWN"]
AMENITIES_FOR_DROPDOWN = bundle["AMENITIES_FOR_DROPDOWN"]
DISTANCE_KM_MEAN = bundle["DISTANCE_KM_MEAN"]

# Amenity score calculation function (copied from notebook preprocessing)
def calculate_amenity_score_app(amenities_str):
    """Calculate a numerical score based on available amenities"""
    if pd.isna(amenities_str) or amenities_str == 'None' or amenities_str == '':
        return 0

    amenities_str = str(amenities_str).lower()

    # Define amenity weights (importance factor)
    amenity_weights = {
        # Premium amenities (high value)
        'swimming': 15,
        'pool': 15,
        'gym': 12,
        'fitness': 12,
        'clubhouse': 10,

        # Security & convenience (medium-high value)
        'gated': 10,
        'security': 10,
        '24hr': 8,
        '24/7': 8,
        'lift': 8,
        'elevator': 8,
        'generator': 7,
        'backup': 7,
        'parking': 6,
        'garage': 6,

        # Comfort amenities (medium value)
        'furnished': 10,
        'semi-furnished': 7,
        'ac': 6,
        'air conditioning': 6,
        'balcony': 5,
        'terrace': 5,
        'garden': 5,
        'landscaped': 4,

        # Basic amenities (low value)
        'wifi': 4,
        'internet': 4,
        'cctv': 4,
        'intercom': 3,
        'water': 3,
        'borehole': 3,
        'staff': 2,
        'quarter': 2,

        # Finishes (premium)
        'modern': 3,
        'luxury': 5,
        'spacious': 2,
        'renovated': 3,
        'new': 2
    }

    score = 0
    amenities_found = []

    for amenity, weight in amenity_weights.items():
        if amenity in amenities_str:
            score += weight
            amenities_found.append(amenity)

    # Bonus: Multiple amenities get synergy bonus
    if len(amenities_found) >= 5:
        score *= 1.1  # 10% bonus for 5+ amenities
    elif len(amenities_found) >= 3:
        score *= 1.05  # 5% bonus for 3+ amenities

    # Cap the score at 100
    return min(round(score), 100)


# ── Header ────────────────────────────────────────────────
st.title("🏡 Nairobi Property Price Predictor")
st.markdown(
    "Enter property details below to get an instant price estimate "
    "powered by a **Random Forest model (R² = 0.93)**." # Updated R^2 based on output
)
st.markdown("---")

# ── Input form ────────────────────────────────────────────
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        location     = st.selectbox("📍 Location", LOCATIONS_FOR_DROPDOWN)
        bedrooms     = st.slider("🛏 Bedrooms", 1, 6, 3)
        bathrooms    = st.slider("🚿 Bathrooms", 1.0, 6.0, 2.0, step=0.5)

    with col2:
        size_sqm     = st.number_input("📐 Size (sqm)", min_value=30.0,
                                       max_value=1000.0, value=150.0, step=10.0)
        # Assuming AMENITIES_FOR_DROPDOWN is a list of unique amenity strings
        selected_amenity_str = st.selectbox("✨ Amenities", AMENITIES_FOR_DROPDOWN)
        distance_km  = st.slider("📏 Distance from CBD (km)", 0.5, 25.0, 5.0, step=0.5)

    submitted = st.form_submit_button("💰 Predict Price", use_container_width=True)

# ── Prediction logic ──────────────────────────────────────
if submitted:
    # Create input DataFrame matching the training features' columns
    # The pipeline expects features in a specific order (NUM first, then CAT for OHE)
    # The input dataframe should have columns matching NUM_FEATURES and CAT_FEATURES.
    input_data = pd.DataFrame({
        'location': [location],
        'property_type': ['Residential'], # Assuming all properties are Residential as per data
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'size_sqm': [size_sqm],
        'amenities': [selected_amenity_str],
        'distance_km': [distance_km],
        'amenity_score': [calculate_amenity_score_app(selected_amenity_str)]
    })

    # Add binary amenity flags to input_data
    for amenity in ALL_AMENITIES:
        col = f"amenity_{amenity.lower().replace('-', '_')}"
        input_data[col] = input_data['amenities'].str.contains(amenity, na=False).astype(int)

    # Ensure the input DataFrame has columns in the expected order (NUM_FEATURES + CAT_FEATURES)
    X_input_df = input_data[NUM_FEATURES + CAT_FEATURES]

    predicted = model.predict(X_input_df)[0]
    lower     = max(0, predicted - MAE)
    upper     = predicted + MAE

    # ── Display results ───────────────────────────────────
    st.markdown("---")
    st.success(f"### 💵 Estimated Price: **KES {predicted:,.0f}**")
    st.info(
        f"📊 **Confidence Range:** KES {lower:,.0f} – KES {upper:,.0f}  "
        f"*(model MAE = KES {MAE:,.0f})*"
    )

    # Plain-language driver explanation
    st.markdown("#### 🔍 What's driving this price?")
    drivers = [
        ("🏠 Size",          f"{size_sqm:.0f} sqm — the single biggest factor (72.2% of model weight)"), # Updated importance
        ("🛏 Bedrooms",      f"{bedrooms} bed{'s' if bedrooms > 1 else ''} — second strongest signal (12.0% of model weight)"), # Updated importance
        ("📍 Location",      f"{location} — Locations like Karen, Nyari, and Lavington often command top-tier premiums."),
        ("📏 Distance",      f"{distance_km} km from CBD — proximity to the CBD significantly impacts price (e.g., Prime 0-5km zone)."),
        ("✨ Amenities",     f"'{selected_amenity_str}' — amenities like 'En-Suite' and 'Gated' features can add significant value."),
    ]
    for label, desc in drivers:
        st.markdown(f"- **{label}:** {desc}")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: Random Forest  |  R² = 0.93  |  MAE ≈ KES 13M  " # Updated R^2 and MAE
    "|  Trained on 49 Nairobi listings  |  Day 5 of 5"
)
