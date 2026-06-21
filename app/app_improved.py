import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Ensure sys.path is correctly set for imports from src/
import sys
root = Path.cwd()   

while root != root.parent:
    if (root / "src").exists():
        break
    root = root.parent

sys.path.insert(0, str(root))

from src.bi_layer import _preprocess_user_input, build_feature_dataframe, predict_price, get_prediction_range, get_shap_explanation, interpret_brand_premium, get_pricing_verdict
from src.recommender import ApplianceRecommender

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Buy: AI Price & Deal Assistant", layout="wide")

# --- 2. HEADER & DESCRIPTION ---
st.title("🛒 Smart Buy: AI Price & Deal Assistant")
st.markdown("""
Find fair market prices and discover the best alternatives for home appliances.
Our AI analyzes prices and recommends similar products within your budget range.
""")

# --- 3. LOADING THE MODELS ---
@st.cache_resource
def load_resources():
    pipeline_path = Path(__file__).parent.parent / "models/saved_models/multi_category_xgb_pipeline.pkl"
    try:
        pipeline = joblib.load(pipeline_path)
        recommender = ApplianceRecommender() 
        return pipeline, recommender
    except Exception as e:
        st.error(f"❌ Failed to load model or recommender: {e}")
        st.stop()

try:
    price_pipeline, recommender_engine = load_resources()
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Critical error during app initialization: {e}")
    st.stop()

# --- 4. HELPER FUNCTIONS FOR DISPLAY ---

def format_recommendation_score(score):
    """
    Turns a cosine similarity score (0 to 1) into a human-readable percentage.
    Shows how well each product matches the user's stated preferences.
    """
    pct = score * 100
    
    if pct >= 90:
        return f"⭐⭐⭐ {pct:.0f}%"
    elif pct >= 75:
        return f"⭐⭐ {pct:.0f}%"
    elif pct >= 60:
        return f"⭐ {pct:.0f}%"
    else:
        return f"{pct:.0f}%"


def display_recommendations(recommendations, category, max_per_brand):
    """
    Displays recommendation results in a clean, professional table.
    Shows recommendations based on feature similarity within the user's budget range.
    """
    if not recommendations:
        st.info("💡 No recommendations found. Try adjusting your preferences or capacity.")
        return

    recs_df = pd.DataFrame(recommendations)

    # Pick which columns to show based on category
    if category == "AC":
        display_cols = ['brand_name', 'capacity_ac_tons', 'star_rating', 'fair_price', 'recommendation_score']
        display_names = ['Brand', 'Capacity (Tons)', '⭐ Rating', 'Fair Price', 'Match Score']
        col_formats = {
            'Fair Price': lambda x: f"₹{x:,.0f}",
            'Match Score': lambda x: format_recommendation_score(x)
        }
    elif category == "Refrigerator":
        display_cols = ['brand_name', 'capacity_ref_liters', 'star_rating', 'fair_price', 'recommendation_score']
        display_names = ['Brand', 'Capacity (L)', '⭐ Rating', 'Fair Price', 'Match Score']
        col_formats = {
            'Fair Price': lambda x: f"₹{x:,.0f}",
            'Match Score': lambda x: format_recommendation_score(x)
        }
    else:  # Washing Machine
        display_cols = ['brand_name', 'capacity_wm_kg', 'star_rating', 'fair_price', 'recommendation_score']
        display_names = ['Brand', 'Capacity (Kg)', '⭐ Rating', 'Fair Price', 'Match Score']
        col_formats = {
            'Fair Price': lambda x: f"₹{x:,.0f}",
            'Match Score': lambda x: format_recommendation_score(x)
        }

    # Defensive check for missing columns
    missing_cols = [c for c in display_cols if c not in recs_df.columns]
    if missing_cols:
        st.error(f"⚠️ Missing columns: {missing_cols}. Check recommender.pool.columns")
        return

    recs_df_display = recs_df[display_cols].copy()
    recs_df_display.columns = display_names

    # Apply formatting
    for col_name, col_display_name in zip(display_cols, display_names):
        if col_display_name == 'Fair Price':
            recs_df_display[col_display_name] = recs_df[col_name].apply(col_formats['Fair Price'])
        elif col_display_name == 'Match Score':
            recs_df_display[col_display_name] = recs_df[col_name].apply(col_formats['Match Score'])

    # Display as a clean, styled dataframe
    st.dataframe(
        recs_df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Brand": st.column_config.TextColumn(width="medium"),
            "Capacity (Tons)": st.column_config.NumberColumn(width="small"),
            "Capacity (L)": st.column_config.NumberColumn(width="small"),
            "Capacity (Kg)": st.column_config.NumberColumn(width="small"),
            "⭐ Rating": st.column_config.NumberColumn(width="small"),
            "Fair Price": st.column_config.TextColumn(width="medium"),
            "Match Score": st.column_config.TextColumn(width="medium"),
        }
    )
    
    st.caption(
        f"✨ Showing {len(recs_df_display)} recommendations | "
        f"Match Score = how well the product matches your preferences (cosine similarity)"
    )


def display_shap_contributions(shap_explanation, category, predicted_price):
    """
    Displays SHAP feature contributions in a clean, readable format.
    Now shows category-specific top contributing features.
    
    IMPROVEMENT: Shows features in rupees with proper sign indicators.
    """
    top_features = shap_explanation['top_features']
    
    # Filter features that are NOT metadata/system features
    # This makes the display more meaningful for users
    exclude_patterns = [
        'onehotencoder__', 'ordinalencoder__', 'standardscaler__',
        'features_above_avg', 'rating_above_avg', 'capacity_above_avg',
        'brand_frequency', 'smart_connectivity_score'
    ]
    
    filtered_features = []
    for feature_name, contribution in top_features:
        # Skip if it matches any exclude pattern
        if any(pattern in feature_name.lower() for pattern in exclude_patterns):
            continue
        filtered_features.append((feature_name, contribution))
    
    # Take top 7 features after filtering
    top_filtered = filtered_features[:7]
    
    if not top_filtered:
        st.warning("⚠️ Could not extract meaningful feature contributions.")
        return
    
    # Create a more readable format
    contribution_data = []
    for feature_name, contribution_rupees in top_filtered:
        # Clean up feature names for display
        display_name = _clean_feature_name(feature_name)
        
        # Determine impact direction
        if contribution_rupees > 0:
            impact = "📈 Increases"
            color = "🟢"
        else:
            impact = "📉 Decreases"
            color = "🔴"
        
        contribution_data.append({
            'Feature': display_name,
            'Impact': impact,
            'Amount (₹)': f"{abs(contribution_rupees):,.0f}",
            '': color
        })
    
    contributions_df = pd.DataFrame(contribution_data)
    
    st.dataframe(
        contributions_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Feature': st.column_config.TextColumn(width="large"),
            'Impact': st.column_config.TextColumn(width="medium"),
            'Amount (₹)': st.column_config.TextColumn(width="medium"),
            '': st.column_config.TextColumn(width="small")
        }
    )


def _clean_feature_name(feature_name: str) -> str:
    """
    Convert technical feature names into human-readable format.
    
    Examples:
    - 'target_enc__brand_name' -> 'Brand Name'
    - 'capacity_ac_tons' -> 'AC Capacity (Tons)'
    - 'ac_split' -> 'Split AC'
    - 'n_features' -> 'Number of Features'
    """
    # Remove transformer prefixes
    clean = feature_name
    prefixes = ['target_enc__', 'ordinalencoder__', 'onehotencoder__', 'standardscaler__']
    for prefix in prefixes:
        if clean.startswith(prefix):
            clean = clean.replace(prefix, '')
    
    # Replace underscores with spaces and title case
    clean = clean.replace('_', ' ').title()
    
    # Specific replacements for better readability
    replacements = {
        'Ac ': 'AC ',
        'Wm ': 'Washing Machine ',
        'Ref ': 'Refrigerator ',
        'N Features': 'Number of Features',
        'Star Rating': 'Star Rating',
        'Brand Name': 'Brand Name',
        'Capacity Ac Tons': 'AC Capacity (Tons)',
        'Capacity Ref Liters': 'Refrigerator Capacity (Liters)',
        'Capacity Wm Kg': 'Washing Machine Capacity (Kg)',
        'Has Inverter': 'Inverter Technology',
        'Has Wifi': 'WiFi Connectivity',
        'Has Voice Control': 'Voice Control',
        'Has App Control': 'App Control',
        'Ac Premium Features': 'AC Premium Features',
        'Ref Door Complexity': 'Refrigerator Door Complexity',
        'Wm Tech Level': 'Washing Machine Tech Level'
    }
    
    for original, readable in replacements.items():
        if original.lower() in clean.lower():
            clean = clean.replace(original, readable)
            break
    
    return clean


# --- 5. USER INPUT SECTION ---
st.header("🔍 Appliance Specifications")

col1, col2 = st.columns(2)

user_friendly_input = {}

with col1:
    # Category selection with user-friendly display names
    category_display = st.selectbox(
        "Select Category",
        ["Air Conditioner", "Refrigerator", "Washing Machine"],
        help="Choose the type of appliance you're interested in"
    )
    
    # Map display names back to internal names used by the model
    category_map = {
        "Air Conditioner": "AC",
        "Refrigerator": "Refrigerator",
        "Washing Machine": "WashingMachine"
    }
    user_friendly_input['category'] = category_map[category_display]
    
    user_friendly_input['brand_name'] = st.text_input(
        "Brand Name",
        value="LG",
        help="e.g., LG, Samsung, Whirlpool, IFB"
    )
    
    # Capacity input with category-specific units
    if category_display == "Air Conditioner":
        user_friendly_input['capacity_value'] = st.number_input(
            "Capacity (Tons)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.5
        )
    elif category_display == "Refrigerator":
        user_friendly_input['capacity_value'] = st.number_input(
            "Capacity (Litres)",
            min_value=100,
            max_value=1000,
            value=250,
            step=10
        )
    else:  # Washing Machine
        user_friendly_input['capacity_value'] = st.number_input(
            "Capacity (Kg)",
            min_value=5,
            max_value=20,
            value=7,
            step=1
        )

with col2:
    user_friendly_input['star_rating'] = st.slider(
        "Star Rating",
        min_value=1,
        max_value=5,
        value=3,
        help="Quality rating (1-5 stars)"
    )
    
    user_friendly_input['has_inverter'] = st.checkbox(
        "Inverter Technology",
        value=True,
        help="More energy efficient"
    )

    # --- CATEGORY-SPECIFIC FEATURES ---
    if category_display == "Air Conditioner":
        st.subheader("AC Features")
        user_friendly_input['ac_split'] = st.checkbox("Split AC", value=True)
        user_friendly_input['ac_copper_condenser'] = st.checkbox("Copper Condenser", value=True)
        user_friendly_input['ac_pm25_filter'] = st.checkbox("PM 2.5 Filter", value=False)
        user_friendly_input['ac_hepa_filter'] = st.checkbox("HEPA Filter", value=False)
        user_friendly_input['ac_auto_clean'] = st.checkbox("Auto Clean", value=False)
        user_friendly_input['ac_hot_and_cold'] = st.checkbox("Hot & Cold", value=False)
        user_friendly_input['ac_Dehumidification'] = st.checkbox("Dehumidification", value=True)
        user_friendly_input['ac_Turbo Mode'] = st.checkbox("Turbo Mode", value=True)
        user_friendly_input['ac_Self Diagnosis'] = st.checkbox("Self Diagnosis", value=False)

    elif category_display == "Refrigerator":
        st.subheader("Refrigerator Features")
        user_friendly_input['ref_frost_free'] = st.checkbox("Frost Free", value=True)
        user_friendly_input['ref_double_door'] = st.checkbox("Double Door", value=True)
        user_friendly_input['ref_multi_door'] = st.checkbox("Multi-Door", value=False)
        user_friendly_input['ref_french_door'] = st.checkbox("French Door", value=False)
        user_friendly_input['ref_convertible'] = st.checkbox("Convertible Freezer", value=False)
        user_friendly_input['ref_door_alarm'] = st.checkbox("Door Alarm", value=True)
        user_friendly_input['ref_door_lock'] = st.checkbox("Door Lock", value=False)
        user_friendly_input['ref_dispenser'] = st.checkbox("Water/Ice Dispenser", value=False)
        user_friendly_input['ref_door_display'] = st.checkbox("Door Display", value=True)

    else:  # Washing Machine
        st.subheader("Washing Machine Features")
        user_friendly_input['wm_fully_automatic'] = st.checkbox("Fully Automatic", value=True)
        user_friendly_input['wm_front_load'] = st.checkbox("Front Load", value=True)
        user_friendly_input['wm_top_load'] = st.checkbox("Top Load", value=False)
        user_friendly_input['wm_with_dryer'] = st.checkbox("Washer Dryer Combo", value=False)
        user_friendly_input['wm_inbuilt_heater'] = st.checkbox("Inbuilt Heater", value=True)
        user_friendly_input['wm_quick_wash'] = st.checkbox("Quick Wash", value=True)
        user_friendly_input['wm_child_lock'] = st.checkbox("Child Lock", value=True)
        user_friendly_input['wm_shock_proof'] = st.checkbox("Shock Proof", value=False)
        user_friendly_input['wm_display'] = st.checkbox("Digital Display", value=True)


# --- 6. MODE SELECTION ---
st.divider()
st.header("🎯 What would you like to do?")
mode = st.radio(
    "Choose an option:",
    ["Explore Fair Prices", "Analyze a Deal"],
    horizontal=True,
    help="Explore to see estimated prices, or Analyze to check if a specific price is a good deal"
)

# --- Observed Price Input for "Analyze a Deal" mode ---
observed_price = None
if mode == "Analyze a Deal":
    observed_price = st.number_input(
        "Enter the price you found (₹)",
        min_value=1000,
        value=30000,
        step=1000,
        help="What price did you see in the market?"
    )

st.divider()

# --- 7. EXECUTION BUTTON ---
if st.button("🔍 Get Insights", use_container_width=True):
    # --- PREPROCESSING USER INPUT ---
    processed_features = _preprocess_user_input(user_friendly_input)
    features_df = build_feature_dataframe(processed_features)

    # --- PRICE PREDICTION (Common to both modes) ---
    predicted_price, _ = predict_price(features_df)
    shap_explanation = get_shap_explanation(features_df)
    brand_premium_text = interpret_brand_premium(shap_explanation)
    lower_bound, upper_bound = get_prediction_range(processed_features)

    if mode == "Explore Fair Prices":
        st.subheader(f"💡 Fair Price Analysis for {category_display}")

        # Display the predicted fair price prominently
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Estimated Fair Price",
                value=f"₹ {predicted_price:,.0f}",
                delta=None
            )
        with col2:
            st.metric(
                label="Price Range",
                value=f"₹ {lower_bound:,.0f}",
                delta=f"to ₹ {upper_bound:,.0f}"
            )
        with col3:
            st.info(f"📊 Based on {category_display} specifications and market data")

        st.write(f"**Brand Impact:** {brand_premium_text}")

        st.subheader("📊 What Affects the Price?")
        display_shap_contributions(shap_explanation, user_friendly_input['category'], predicted_price)
        st.caption("Shows which features have the most impact on pricing for your selected appliance")

        st.subheader("⭐ Smart Recommendations (5 Best Matches)")
        st.write("Based on your preferences, within your expected budget range:")
        recommendations = recommender_engine.get_recommendations(user_friendly_input, n=5)
        display_recommendations(recommendations, user_friendly_input['category'], recommender_engine.max_per_brand)

    else:  # mode == "Analyze a Deal"
        st.subheader(f"💰 Is This a Good Deal? {category_display}")

        if observed_price is None:
            st.warning("👆 Please enter a price above to analyze.")
        else:
            deal_verdict = get_pricing_verdict(observed_price=observed_price, predicted_price=predicted_price)

            # Display key metrics side by side
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Price You Found",
                    value=f"₹ {observed_price:,.0f}"
                )
            with col2:
                st.metric(
                    label="Fair Price",
                    value=f"₹ {predicted_price:,.0f}"
                )
            with col3:
                difference = observed_price - predicted_price
                if difference < 0:
                    st.metric(
                        label="Potential Savings",
                        value=f"₹ {abs(difference):,.0f}",
                        delta="Good deal! ✅"
                    )
                else:
                    st.metric(
                        label="Potential Overpayment",
                        value=f"₹ {difference:,.0f}",
                        delta="Overpriced ⚠️"
                    )

            st.write(f"*(Expected fair price range: ₹{lower_bound:,.0f} - ₹{upper_bound:,.0f})*")

            # Display verdict with color coding
            if deal_verdict['verdict'] in ["Exceptional Value", "Good Deal"]:
                st.success(f"🎉 **{deal_verdict['verdict']}** — {deal_verdict['message']}")
            elif deal_verdict['verdict'] == "Fairly Priced":
                st.info(f"👍 **{deal_verdict['verdict']}** — {deal_verdict['message']}")
            else:
                st.warning(f"⚠️ **{deal_verdict['verdict']}** — {deal_verdict['message']}")

            st.write(f"**Brand Impact:** {brand_premium_text}")

            st.subheader("📊 What Affects the Price?")
            display_shap_contributions(shap_explanation, user_friendly_input['category'], predicted_price)
            st.caption("Shows which features have the most impact on pricing for your selected appliance")

            st.subheader("⭐ Better Alternatives (5 Smart Recommendations)")
            st.write("Found a better option within your budget range:")
            recommendations = recommender_engine.get_recommendations(user_friendly_input, n=5)
            display_recommendations(recommendations, user_friendly_input['category'], recommender_engine.max_per_brand)
