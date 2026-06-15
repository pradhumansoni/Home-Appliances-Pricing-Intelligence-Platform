# Stage 1: The Setup & Inputs 

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.bi_layer import _preprocess_user_input, build_feature_dataframe
from src.recommender import Recommender

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Buy: AI Price & Deal Assistant", layout="wide")

st.title("🛒 Smart Buy: AI Price & Deal Assistant")
st.markdown("Analyze the best deals or explore fair market prices for home appliances.")

# --- 2. LOADING THE MODELS ---
@st.cache_resource
def load_resources():
    # Load your EXISTING pipeline (preprocessor + model)
    pipeline_path = Path("models/saved_models/multi_category_xgb_pipeline.pkl")
    pipeline = joblib.load(pipeline_path)
    
    # Initialize Recommender (which loads X_train.parquet internally)
    recommender = Recommender() 
    return pipeline, recommender

try:
    price_pipeline, recommender_engine = load_resources()
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")
    st.stop()  # Stop the app if models fail to load

# --- 3. USER INPUT SECTION ---
st.header("🔍 Appliance Specifications")

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Select Category", ["AC", "Refrigerator", "Washing Machine"])
    brand = st.text_input("Enter Brand Name (e.g., LG, Samsung, Whirlpool)", "LG")
    
    # Dynamic capacity input based on category
    if category == "AC":
        capacity = st.number_input("Capacity in Tons", min_value=0.5, max_value=3.0, value=1.5, step=0.5)
    elif category == "Refrigerator":
        capacity = st.number_input("Capacity in Litres", min_value=100, max_value=1000, value=250, step=10)
    else:  # Washing Machine
        capacity = st.number_input("Capacity in Kg", min_value=5, max_value=20, value=7, step=1)

with col2:
    star_rating = st.slider("Star Rating", 1, 5, 3)
    
    # Category-specific features
    if category == "AC":
        has_inverter = st.checkbox("Inverter AC", value=True)
        ac_split = st.checkbox("Split AC", value=True)
    elif category == "Refrigerator":
        has_inverter = st.checkbox("Inverter Compressor", value=True)
        ref_frost_free = st.checkbox("Frost Free", value=True)
    else:  # Washing Machine
        has_inverter = st.checkbox("Inverter Motor", value=True)
        wm_fully_automatic = st.checkbox("Fully Automatic", value=True)

# --- 4. MODE SELECTION ---
st.divider()
st.header("🎯 Choose Your Goal")
mode = st.radio(
    "What would you like to do?",
    ["Explore Fair Prices", "Analyze a Deal"],
    horizontal=True
)

st.info(f"Currently in **{mode}** mode. (Logic coming in Stage 2 & 3)")