# 1. Load the model and extract preprocessor, pipeline and feature names from it

"""
(I)
Business Intelligence Layer for Appliance Pricing Intelligence Platform.

This module converts model predictions into actionable consumer insights.
It does NOT modify predictions - it interprets them.
"""

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Model path
MODEL_PATH = Path(__file__).parent.parent / "models/saved_models/final_model.pkl"
# Path to the training features DataFrame

# We need this to derive default values for missing user inputs.
TRAIN_FEATURES_PATH = Path(__file__).parent.parent / "models/dataset/X_train.parquet"

# Load the trained model pipeline
_pipeline = None
_preprocessor = None
_model = None
_feature_names = None

# Store the original order of features from the training data

_original_feature_order: Optional[list] = None

# Store the default feature values in global variable 

_default_feature_values: Optional[Dict[str, Any]] = None

# NEW: Store the data types of the original training features
_original_feature_dtypes: Optional[Dict[str, Any]] = None


def _load_model():
    """Load the model pipeline and extract components."""
    global _pipeline, _preprocessor, _model, _feature_names
    
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
        _preprocessor = _pipeline.named_steps['preprocessor']
        _model = _pipeline.named_steps['model']
        _feature_names = _preprocessor.get_feature_names_out()
    
    return _pipeline, _preprocessor, _model, _feature_names

def _load_default_feature_values():
    """
    Load the training feature DataFrame, compute default (mode/median) values,
    store the original feature order, and store original feature data types.
    """
    global _default_feature_values, _original_feature_order, _original_feature_dtypes

    if _default_feature_values is None:
        try:
            # print(f"[DEBUG] Attempting to load X_train from: {TRAIN_FEATURES_PATH}") # Removed debug print for clarity
            X_train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
            # print(f"[DEBUG] X_train_df loaded successfully. Shape: {X_train_df.shape}") # Removed debug print

            _default_feature_values = {}
            _original_feature_dtypes = {} # Initialize dtypes dictionary
            for col in X_train_df.columns:
                _original_feature_dtypes[col] = X_train_df[col].dtype # Store dtype

                if pd.api.types.is_numeric_dtype(X_train_df[col]):
                    _default_feature_values[col] = X_train_df[col].median()
                else:
                    _default_feature_values[col] = X_train_df[col].mode().iloc[0]

            if 'log_price' in _default_feature_values:
                del _default_feature_values['log_price']
            if '__index_level_0__' in _default_feature_values: # Ensure index is not a feature
                del _default_feature_values['__index_level_0__']
            if '__index_level_0__' in _original_feature_dtypes: # Remove index dtype too
                del _original_feature_dtypes['__index_level_0__']

            # Store the exact column order from the training DataFrame, excluding __index_level_0__
            _original_feature_order = [col for col in X_train_df.columns.tolist() if col != '__index_level_0__']
            # print(f"[DEBUG] Default feature values, original order, and dtypes set. Number of defaults: {len(_default_feature_values)}") # Removed debug print

        except Exception as e:
            # print(f"[ERROR] Failed to load default feature values: {e}") # Removed debug print
            _default_feature_values = None
            _original_feature_order = None
            _original_feature_dtypes = None # Also clear dtypes on error
            raise # Re-raise the exception

# Load model on module import
_load_model()
# print("[DEBUG] _load_model called.") # Removed debug print

# Call the function to load default feature values on module import
_load_default_feature_values()
# print("[DEBUG] _load_default_feature_values called during module import.") # Removed debug print


# --------------------------------------------------------------------------------------------------------------------

# 2. Function to build a complete feature DataFrame from partial user input

def build_feature_dataframe(user_features: Dict[str, Any]) -> pd.DataFrame:
    """
    Constructs a complete feature DataFrame for prediction, filling missing
    user inputs with default values derived from the training data.

    Args:
        user_features: A dictionary of features provided by the user.
                       Can be incomplete.

    Returns:
        A pandas DataFrame with all expected features, ready for the model.
        Missing features are filled with defaults.
    """
    if _default_feature_values is None or _original_feature_order is None:
        raise RuntimeError("Default feature values or original feature order not loaded. "
                           "Ensure _load_default_feature_values() was called successfully.")

    # Start with a copy of the global default feature values.
    # We copy it to avoid modifying the global dictionary directly.
    complete_features = _default_feature_values.copy()

    # Debug: Print the default features for comparison
    # print(f"[DEBUG BI] Default features for comparison: {complete_features}")

    # Iterate through the user-provided features and update our complete_features dictionary.
    # User-provided values will overwrite the defaults.
    for key, value in user_features.items():
        # We only update if the key exists in our expected features
        # and the value is not None or an empty string, implying it's a valid input.
        if key in complete_features and value is not None and str(value).strip() != '':
            complete_features[key] = value

    # Convert the complete_features dictionary into a pandas DataFrame.
    # We wrap it in a list `[complete_features]` to create a single-row DataFrame.
    # The `columns=_original_feature_order` ensures that the DataFrame's columns
    # are in the same order as the training data, which is critical for the model's preprocessor.
    features_df = pd.DataFrame([complete_features], columns=_original_feature_order)

    # Debug: Print the final DataFrame's relevant columns
    # print(f"[DEBUG BI] Final DataFrame head for prediction:\n{features_df.head()}")


    return features_df


#3. Add Prediction Function

def predict_price(features_df: pd.DataFrame) -> Tuple[float, np.ndarray]:
    """
    Predict the fair price for given features.
    
    Args:
        features_df: DataFrame with appliance features (must match training format)
    
    Returns:
        Tuple of (predicted_price_rupees, log_prediction)
    """
    # Get log-space prediction
    log_pred = _pipeline.predict(features_df)[0]
    
    # Convert back to rupee space
    price = np.expm1(log_pred)
    
    return price, log_pred

# ------------------------------------------------------------------------------------------------------------------------------------

# 4. Add Pricing Verdict Function - CORRECTED FOR SAVINGS LOGIC
def get_pricing_verdict(observed_price: float, predicted_price: float) -> Dict[str, Any]:
    """
    Determine pricing verdict based on observed vs predicted price.

    Args:
        observed_price: The market price you observed.
        predicted_price: The fair price predicted by the model.

    Returns:
        Dictionary containing verdict, difference, percentage,
        consumer_advantage (if observed < predicted),
        consumer_disadvantage (if observed > predicted),
        message, predicted_price, and observed_price.
    """

    difference = observed_price - predicted_price
    percentage = (difference / predicted_price) * 100

    # Initialize message
    message = ""

    if abs(percentage) <= 10:
        verdict = "Fairly Priced"
        if difference > 0:
            message = (
                f"The price is ₹{abs(difference):,.0f} above the estimated fair value "
                f"but still within the normal pricing range."
            )
        elif difference < 0:
            message = (
                f"The price is ₹{abs(difference):,.0f} below the estimated fair value "
                f"but still within the normal pricing range."
            )
        else:
            message = "The observed price matches the estimated fair value."

    elif 10 < percentage <= 20:
        verdict = "Slightly Overpriced"
        message = f"You may be slightly overpaying by ₹{difference:,.0f}."

    elif percentage > 20:
        verdict = "Overpriced"
        message = f"You may be significantly overpaying by ₹{difference:,.0f}."

    elif -20 <= percentage < -10:
        verdict = "Good Deal"
        message = (
            f"Estimated savings of ₹{abs(difference):,.0f} "
            f"compared to the model's fair value (a good deal!)."
        )

    else:  # percentage < -20
        verdict = "Exceptional Value"
        message = (
            f"This is an exceptional value! Estimated savings of ₹{abs(difference):,.0f} "
            f"compared to the model's fair value."
        )

    return {
        "verdict": verdict,
        "difference": difference,
        "percentage": percentage,
        "consumer_advantage": max(0, -difference),    # How much cheaper it is than predicted (a positive value)
        "consumer_disadvantage": max(0, difference), # How much more expensive it is than predicted (a positive value)
        "message": message,
        "predicted_price": predicted_price,
        "observed_price": observed_price
    }

# --------------------------------------------------------------------------------------------------------------------------------------

# 5. SHAP Explanation Function

def get_shap_explanation(features_df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
    """
    Get SHAP explanation for a prediction.
    
    Args:
        features_df: DataFrame with appliance features
        top_n: Number of top features to return
    
    Returns:
        Dictionary with base value,shap values, and feature contributions
    """
    # Transform features through the preprocessing pipeline
    X_transformed = _preprocessor.transform(features_df)
    
    # Convert to dense array if sparse
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    
    X_transformed_df = pd.DataFrame(X_transformed, columns=_feature_names).astype(np.float64)
    
    # Initialize SHAP explainer for tree models
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_transformed_df.iloc[[0]])
    
    # Get feature contributions
    feature_contributions = list(
    zip(_feature_names, shap_values[0])
)
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        "base_value": explainer.expected_value,
        "shap_values": shap_values,
        "top_features": feature_contributions[:top_n]
    }

# -------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Add Brand Premium Interpretation

def interpret_brand_premium(shap_explanation: Dict, brand_feature_prefix: str = "target_enc__brand_name") -> str:
    """
    Interpret brand premium from SHAP values.
    
    Args:
        shap_explanation: Output from get_shap_explanation()
        brand_feature_prefix: The brand encoding feature name pattern
    
    Returns:
        Human-readable interpretation of brand premium
    """
    top_features = shap_explanation["top_features"]
    
    # Find brand-related features
    brand_effects = [(name, val) for name, val in top_features 
                     if brand_feature_prefix in name]
    
    if not brand_effects:
        return "Brand effect could not be isolated from other factors."
    
    # Sum brand effects
    total_brand_effect = sum(val for _, val in brand_effects)

    baseline_price = np.expm1(shap_explanation["base_value"])

    price_with_brand = np.expm1(
        shap_explanation["base_value"] + total_brand_effect
    )

    brand_effect_rupees = (
        price_with_brand - baseline_price
    )   
    return (f"About ₹{abs(brand_effect_rupees):,.0f} of the estimated price "
            f"is attributable to brand effects. "
            f"{'Premium' if brand_effect_rupees > 0 else 'Budget'} brands tend to command this adjustment.")
#-------------------------------------------------------------------------------------------------------------------------------------------------------


# NEW: Helper to safely convert to a binary (0/1) integer
def _to_binary_int(value: Any) -> int:
    """Converts various inputs to a 0 or 1 integer."""
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ['true', 'yes', '1', 'on']:
            return 1
        elif value_lower in ['false', 'no', '0', 'off']:
            return 0
    elif isinstance(value, (bool, int, float)):
        return 1 if value else 0
    return 0 # Default to 0 if unsure or invalid

# NEW: Function to preprocess user-friendly input into model-friendly features
def _preprocess_user_input(user_friendly_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes user-friendly input and transforms it into model-ready features,
    including standardization, type coercion, and basic feature derivation.

    Args:
        user_friendly_features: A dictionary of features provided by the user
                                in a simplified, human-readable format.

    Returns:
        A dictionary with features mapped to the model's expected nomenclature
        and data types, ready for build_feature_dataframe.
    """
    if _original_feature_dtypes is None:
        raise RuntimeError("Original feature data types not loaded. Ensure _load_default_feature_values() was called.")

    processed_features = {}

    # 1. Handle 'category' and 'brand_name' (lowercase)
    category = user_friendly_features.get('category', '').lower().strip()
    processed_features['category'] = category # Store processed category
    if 'brand_name' in user_friendly_features:
        processed_features['brand_name'] = str(user_friendly_features['brand_name']).lower().strip()

    # 2. Handle 'rating' (float) and 'star_rating' (float)
    if 'rating' in user_friendly_features and user_friendly_features['rating'] is not None:
        try:
            processed_features['rating'] = float(user_friendly_features['rating'])
        except ValueError:
            pass # Default will be used

    if 'star_rating' in user_friendly_features and user_friendly_features['star_rating'] is not None:
        try:
            processed_features['star_rating'] = float(user_friendly_features['star_rating'])
        except ValueError:
            pass # Default will be used

    # 3. Handle Generic Capacity (capacity_value) and map to specific 'capacity_X'
    capacity_value = user_friendly_features.get('capacity_value')

    if capacity_value is not None:
        try:
            capacity_value = float(capacity_value)
            # Use category to map to the correct capacity column
            if category == 'ac': # Use 'ac' as category from preprocessed value
                processed_features['capacity_ac_tons'] = capacity_value
            elif category == 'refrigerator':
                processed_features['capacity_ref_liters'] = capacity_value
            elif category == 'washing machine':
                processed_features['capacity_wm_kg'] = capacity_value
        except ValueError:
            pass # Default will be used if conversion fails

    # 4. Handle common binary/boolean features (map to 0 or 1)
    # Check against _original_feature_dtypes to ensure we only process expected features
    for key in ['has_inverter', 'has_wifi', 'has_voice_control', 'has_app_control']: # Added other smart features
        if key in user_friendly_features and key in _original_feature_dtypes:
            processed_features[key] = _to_binary_int(user_friendly_features[key])
            
    # Also handle has_star_rating if star_rating is provided
    if 'star_rating' in processed_features and processed_features['star_rating'] > 0:
        processed_features['has_star_rating'] = 1
    else:
        processed_features['has_star_rating'] = 0


    # 5. Handle specific feature flags based on category (user-friendly string to 0/1)
    # Iterate through ALL user_friendly_features and map them if they exist in _original_feature_dtypes
    
    # --- AC Features ---
    if category == 'ac':
        for key in [
            'ac_split', 'ac_window', 'ac_pm25_filter', 'ac_hepa_filter', 
            'ac_auto_clean', 'ac_hot_and_cold', 'ac_copper_condenser', 
            'ac_Dehumidification', 'ac_Turbo Mode', 'ac_Self Diagnosis'
        ]:
            if key in user_friendly_features and key in _original_feature_dtypes:
                processed_features[key] = _to_binary_int(user_friendly_features[key])
    
    # --- Refrigerator Features ---
    elif category == 'refrigerator':
        for key in [
            'ref_single_door', 'ref_multi_door', 'ref_chest_freezer', 
            'ref_side_door', 'ref_french_door', 'ref_double_door', 
            'ref_triple_door', 'ref_frost_free', 'ref_convertible', 
            'ref_door_alarm', 'ref_door_lock', 'ref_dispenser', 
            'ref_door_display', 'ref_mini'
        ]:
            if key in user_friendly_features and key in _original_feature_dtypes:
                processed_features[key] = _to_binary_int(user_friendly_features[key])
        
        # Explicitly handle door_type if you plan to use it as a single input
        # if 'door_type' in user_friendly_features:
        #     door_type_lower = user_friendly_features['door_type'].lower().strip()
        #     if door_type_lower == 'single door' and 'ref_single_door' in _original_feature_dtypes: processed_features['ref_single_door'] = 1
        #     elif door_type_lower == 'double door' and 'ref_double_door' in _original_feature_dtypes: processed_features['ref_double_door'] = 1
        #     # Add other door types (ensure only one is 1)

    # --- Washing Machine Features ---
    elif category == 'washing machine':
        for key in [
            'wm_fully_automatic', 'wm_semi_automatic', 'wm_with_dryer', 
            'wm_washer_only', 'wm_dryer_only', 'wm_front_load', 
            'wm_top_load', 'wm_inbuilt_heater', 'wm_quick_wash', 
            'wm_ss_tub', 'wm_child_lock', 'wm_shock_proof', 
            'wm_display'
        ]:
            if key in user_friendly_features and key in _original_feature_dtypes:
                processed_features[key] = _to_binary_int(user_friendly_features[key])
        
        # Explicitly handle load_type/wash_type if you plan to use it as a single input
        # if 'load_type' in user_friendly_features:
        #     load_type_lower = user_friendly_features['load_type'].lower().strip()
        #     if load_type_lower == 'front load' and 'wm_front_load' in _original_feature_dtypes: processed_features['wm_front_load'] = 1
        #     elif load_type_lower == 'top load' and 'wm_top_load' in _original_feature_dtypes: processed_features['wm_top_load'] = 1


    # 6. Basic Derivation for 'n_features' (CORRECTED)
    # This should count how many 1s are in the feature flags for the current category
    # It also needs to count general features like has_inverter, has_star_rating
    
    initial_n_features = 0
    # Count general binary features
    for key in ['has_inverter', 'has_star_rating', 'has_wifi', 'has_voice_control', 'has_app_control']:
        if key in processed_features and processed_features[key] == 1:
            initial_n_features += 1

    # Count category-specific binary features
    if category == 'ac':
        for key in [
            'ac_split', 'ac_window', 'ac_pm25_filter', 'ac_hepa_filter', 
            'ac_auto_clean', 'ac_hot_and_cold', 'ac_copper_condenser', 
            'ac_Dehumidification', 'ac_Turbo Mode', 'ac_Self Diagnosis'
        ]:
            if key in processed_features and processed_features[key] == 1:
                initial_n_features += 1
    elif category == 'refrigerator':
        for key in [
            'ref_single_door', 'ref_multi_door', 'ref_chest_freezer', 
            'ref_side_door', 'ref_french_door', 'ref_double_door', 
            'ref_triple_door', 'ref_frost_free', 'ref_convertible', 
            'ref_door_alarm', 'ref_door_lock', 'ref_dispenser', 
            'ref_door_display', 'ref_mini'
        ]:
            if key in processed_features and processed_features[key] == 1:
                initial_n_features += 1
    elif category == 'washing machine':
        for key in [
            'wm_fully_automatic', 'wm_semi_automatic', 'wm_with_dryer', 
            'wm_washer_only', 'wm_dryer_only', 'wm_front_load', 
            'wm_top_load', 'wm_inbuilt_heater', 'wm_quick_wash', 
            'wm_ss_tub', 'wm_child_lock', 'wm_shock_proof', 
            'wm_display'
        ]:
            if key in processed_features and processed_features[key] == 1:
                initial_n_features += 1

    processed_features['n_features'] = initial_n_features


    # 7. Derivations for Engineered Features (If they exist in _original_feature_dtypes)
    # These typically use values that are now in processed_features
    # We should only calculate these if the base components exist
    
    # Capacity Squared
    if 'capacity_ac_tons' in processed_features:
        processed_features['capacity_sq'] = processed_features['capacity_ac_tons'] ** 2
    elif 'capacity_ref_liters' in processed_features:
        processed_features['capacity_sq'] = processed_features['capacity_ref_liters'] ** 2
    elif 'capacity_wm_kg' in processed_features:
        processed_features['capacity_sq'] = processed_features['capacity_wm_kg'] ** 2
    # Ensure n_features_sq is calculated if n_features is present
    if 'n_features' in processed_features:
        processed_features['n_features_sq'] = processed_features['n_features'] ** 2
    if 'star_rating' in processed_features: # assuming rating is same as star_rating
        processed_features['rating_sq'] = processed_features['star_rating'] ** 2

    # Interaction terms
    if 'capacity_ac_tons' in processed_features and 'n_features' in processed_features:
        processed_features['capacity_n_features'] = processed_features['capacity_ac_tons'] * processed_features['n_features']
    elif 'capacity_ref_liters' in processed_features and 'n_features' in processed_features:
        processed_features['capacity_n_features'] = processed_features['capacity_ref_liters'] * processed_features['n_features']
    elif 'capacity_wm_kg' in processed_features and 'n_features' in processed_features:
        processed_features['capacity_n_features'] = processed_features['capacity_wm_kg'] * processed_features['n_features']

    if 'capacity_ac_tons' in processed_features and 'star_rating' in processed_features:
        processed_features['capacity_rating'] = processed_features['capacity_ac_tons'] * processed_features['star_rating']
    elif 'capacity_ref_liters' in processed_features and 'star_rating' in processed_features:
        processed_features['capacity_rating'] = processed_features['capacity_ref_liters'] * processed_features['star_rating']
    elif 'capacity_wm_kg' in processed_features and 'star_rating' in processed_features:
        processed_features['capacity_rating'] = processed_features['capacity_wm_kg'] * processed_features['star_rating']
        
    if 'star_rating' in processed_features and 'n_features' in processed_features:
        processed_features['rating_n_features'] = processed_features['star_rating'] * processed_features['n_features']

    # Smart Intensity (Sum of smart features)
    smart_intensity = 0
    for key in ['has_wifi', 'has_voice_control', 'has_app_control']:
        if key in processed_features and processed_features[key] == 1:
            smart_intensity += 1
    processed_features['smart_intensity'] = smart_intensity

    # Category-specific premium scores (if they exist in _original_feature_dtypes)
    if category == 'ac':
        processed_features['ac_premium_features'] = sum(
            processed_features.get(k, 0) for k in [
                'ac_pm25_filter', 'ac_hepa_filter', 'ac_auto_clean', 
                'ac_hot_and_cold', 'ac_copper_condenser', 'ac_Dehumidification', 
                'ac_Turbo Mode', 'ac_Self Diagnosis'
            ] if k in processed_features
        )
    elif category == 'refrigerator':
        processed_features['ref_door_complexity'] = sum(
            processed_features.get(k, 0) for k in [
                'ref_multi_door', 'ref_french_door', 'ref_double_door', 
                'ref_triple_door', 'ref_door_display'
            ] if k in processed_features
        )
    elif category == 'washing machine':
        processed_features['wm_tech_level'] = sum(
            processed_features.get(k, 0) for k in [
                'wm_fully_automatic', 'wm_with_dryer', 'wm_front_load', 
                'wm_inbuilt_heater', 'wm_quick_wash', 'wm_child_lock', 
                'wm_display'
            ] if k in processed_features
        )
    
    # Statistical features like 'features_above_avg' etc. usually require global stats from X_train
    # and are hard to calculate purely from user input. build_feature_dataframe will use defaults for these.

    # Debug: Print the processed features before returning
    # print(f"[DEBUG BI] Processed features (before build_feature_dataframe): {processed_features}")
    
    return processed_features


# --------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Fair Price Estimation Function - UPDATED with partial features input
def get_fair_price(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get fair price estimate with explanation, handling partial user inputs.

    Args:
        features: Dictionary of appliance features provided by the user.
                  Can be incomplete; missing features will be filled with defaults.

    Returns:
        Dictionary with price, explanation, and brand premium.
    """
    # First, build the complete feature DataFrame from the user's potentially partial input.
    features_df = build_feature_dataframe(features)

    # Get prediction
    predicted_price, log_pred = predict_price(features_df)

    # Get SHAP explanation
    shap_exp = get_shap_explanation(features_df)

    # Get brand premium interpretation
    brand_premium = interpret_brand_premium(shap_exp)

    return {
        "fair_price": predicted_price,
        "explanation": shap_exp,
        "brand_premium": brand_premium
    }

# -----------------------------------------------------------------------------------------------------------------

#7. Add Prediction Range Function

def get_prediction_range(features: Dict[str, Any], n_samples: int = 100) -> Tuple[float, float]:
    """
    Estimate prediction confidence interval using bootstrap-like sampling.
    
    Note: This is a simplified approach. For production, consider:
    - Using model's inherent uncertainty (e.g., quantile regression)
    - Proper bootstrap sampling from training data
    
    Args:
        features: Dictionary of appliance features
        n_samples: Number of samples for estimation
    
    Returns:
        Tuple of (lower_bound, upper_bound) in rupees
    """
    features_df = build_feature_dataframe(features)
    predicted_price, _ = predict_price(features_df)
    
    # Simplified: use a percentage-based range
    # In production, this would use proper uncertainty quantification
    margin = predicted_price * 0.15  # 15% margin
    
    return (predicted_price - margin, predicted_price + margin)




