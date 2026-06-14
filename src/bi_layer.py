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
MODEL_PATH = Path(__file__).parent.parent / "models/saved_models/multi_category_xgb_pipeline.pkl"
# Load the trained model pipeline
_pipeline = None
_preprocessor = None
_model = None
_feature_names = None

def _load_model():
    """Load the model pipeline and extract components."""
    global _pipeline, _preprocessor, _model, _feature_names
    
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
        _preprocessor = _pipeline.named_steps['preprocessor']
        _model = _pipeline.named_steps['model']
        _feature_names = _preprocessor.get_feature_names_out()
    
    return _pipeline, _preprocessor, _model, _feature_names

# Load model on module import
_load_model()

# --------------------------------------------------------------------------------------------------------------------

#2. Add Prediction Function

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

# 3. Add Pricing Verdict Function

def get_pricing_verdict(observed_price: float, predicted_price: float) -> Dict[str, Any]:
    """
    Determine pricing verdict based on observed vs predicted price.

    Args:
        observed_price: The market price you observed.
        predicted_price: The fair price predicted by the model.

    Returns:
        Dictionary containing verdict, difference, percentage,
        savings, and an explanation message.
    """

    difference = observed_price - predicted_price
    percentage = (difference / predicted_price) * 100

    if percentage > 10:
        # Significantly above fair value
        verdict = "Overpriced"
        savings = difference
        message = f"You may be overpaying by ₹{savings:,.0f}."

    elif percentage < -10:
        # Significantly below fair value
        verdict = "Good Deal"
        savings = abs(difference)
        message = (
            f"Estimated savings of ₹{savings:,.0f} "
            f"compared to the model's fair value."
        )

    else:
        # Within ±10% of fair value
        verdict = "Fairly Priced"
        savings = abs(difference)

        if difference > 0:
            message = (
                f"The price is ₹{savings:,.0f} above the estimated fair value "
                f"but still within the normal pricing range."
            )
        elif difference < 0:
            message = (
                f"The price is ₹{savings:,.0f} below the estimated fair value "
                f"but still within the normal pricing range."
            )
        else:
            savings = 0
            message = "The observed price matches the estimated fair value."

    return {
    "verdict": verdict,
    "difference": difference,
    "percentage": percentage,
    "consumer_advantage": max(0, -difference),
    "consumer_disadvantage": max(0, difference),
    "message": message,
    "predicted_price": predicted_price,
    "observed_price": observed_price
}

# --------------------------------------------------------------------------------------------------------------------------------------

# 4. SHAP Explanation Function

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

# 5. Add Brand Premium Interpretation

def interpret_brand_premium(shap_explanation: Dict, brand_feature_prefix: str = "target_enc_brand_name") -> str:
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

# --------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Fair Price Estimation Function - UPDATED with partial features input
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
    features_df = pd.DataFrame([features])
    predicted_price, _ = predict_price(features_df)
    
    # Simplified: use a percentage-based range
    # In production, this would use proper uncertainty quantification
    margin = predicted_price * 0.15  # 15% margin
    
    return (predicted_price - margin, predicted_price + margin)

# ----------------------------------------------------------------------------------------------------------------------
#                   PARTIAL INPUT INFERENCE
#-----------------------------------------------------------------------------------------------------------------------

'''
- To implement this, we need to modify how we prepare the input features before passing them to the model.
- We have to define a BASE or DEFAULT feature set based on the distribution of that feature.
When a user does not provide that specific feature details, we will use the default ones.
'''

# 1. Function to load training data and compute default feature values

# Store the original order of features from the training data

_original_feature_order: Optional[list] = None

# Store the default feature values in global variable 

_default_feature_values: Optional[Dict[str, Any]] = None

# Path to the training features DataFrame
# We need this to derive default values for missing user inputs.
TRAIN_FEATURES_PATH = Path(__file__).parent.parent / "models/dataset/X_train.parquet"

def _load_default_feature_values():
    """Load the training feature DataFrame and compute default (mode/median) values."""
    global _default_feature_values

    if _default_feature_values is None:
        # Load the training features dataset
        # This DataFrame contains all the columns (features) that the model was trained on.
        X_train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
        # Initialize an empty dictionary to store the default values
        _default_feature_values = {}

        # Iterate through each column (feature) in the training data
        for col in X_train_df.columns:
            # Check the data type of the column
            if pd.api.types.is_numeric_dtype(X_train_df[col]): # more safe way
                # If it's a numeric column, use the median as the default value.
                # Median is more robust to outliers than the mean.
                _default_feature_values[col] = X_train_df[col].median()
            else:
                # If it's a categorical (object, string, boolean, etc.) column, use the mode.
                # Mode finds the most frequent value. .iloc[0] is used because mode() can return multiple values if there's a tie.
                _default_feature_values[col] = X_train_df[col].mode().iloc[0]
        
        # Special handling for 'log_price' if it somehow ended up in X_train (it shouldn't, but as a safeguard)
        if 'log_price' in _default_feature_values:
            del _default_feature_values['log_price'] # Ensure target is not a feature

        # Store the exact column order from the training DataFrame
        # This is vital for ensuring consistency when creating new DataFrames
        # from user input, as the preprocessor expects features in a specific order.
        _original_feature_order = X_train_df.columns.tolist()

# Call the function to load default feature values on module import
# This ensures that _default_feature_values is populated when bi_layer.py is first imported.
_load_default_feature_values()

# ---------------------------------------------------------------------------------------------------------------------------

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

    return features_df


# ... (existing predict_price function) ...
# Note: The existing predict_price function still expects a full DataFrame.
# We will integrate build_feature_dataframe into get_fair_price later.

# ... (rest of your existing code for get_pricing_verdict, get_shap_explanation, etc.) ...

