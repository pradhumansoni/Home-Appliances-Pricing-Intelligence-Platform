import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any
import joblib

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]

# Add project root to Python path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# Internal imports from your BI layer
from src.bi_layer import build_feature_dataframe, _load_model

# Relative file loading
X_train_path = BASE_DIR / "models" / "dataset" / "X_train.parquet"
y_train_path = BASE_DIR / "models" / "dataset" / "y_train.parquet"

class ApplianceRecommender:
    def __init__(self):
        # 1. Load Model and Data
        self.pipeline, self.preprocessor, self.model, _ = _load_model()
        
        self.X_train = pd.read_parquet(X_train_path)
        self.y_train = pd.read_parquet(y_train_path)
        
        # 2. Pre-calculate the 'Market Pool'
        self.pool = self.X_train.copy()
        self.pool['actual_price'] = np.expm1(self.y_train['log_price'])
        
        # 3. Calculate Fair Prices for the pool to find "Deals"
        # We do this once at startup
        log_preds = self.pipeline.predict(self.X_train)
        self.pool['fair_price'] = np.expm1(log_preds)
        self.pool['value_score'] = self.pool['fair_price'] / self.pool['actual_price']

    def get_recommendations(self, user_input: Dict[str, Any], n=10) -> List[Dict]:
        category = user_input.get('category')
        capacity = user_input.get('capacity_value')
        
        # Map category to the specific capacity column in your pool
        category_capacity_map = {
            'AC': 'capacity_ac_tons',
            'Refrigerator': 'capacity_ref_litres',
            'Washing Machine': 'capacity_wm_kg'
        }
        
        # Get the correct column name based on the category
        capacity_col = category_capacity_map.get(category)
        
        if not capacity_col:
            return [] # Invalid category

        # 1. HARD CONSTRAINTS (The "Deal-Breakers")
        # Hard Constraint: Filter by category and the SPECIFIC capacity column
        mask = (self.pool['category'] == category) & (self.pool[capacity_col] == capacity)
        eligible_pool = self.pool[mask].copy()
            
        if eligible_pool.empty:
            return []

        # 2. SOFT CONSTRAINTS (The "Preferences")
        # We handle Brand, Star Rating, and Inverter via Similarity Math.
        # If the user mentioned 'lg', the cosine_similarity will naturally 
        # give higher scores to LG products, but it will still show others.
        
        user_encoded = self.preprocessor.transform(build_feature_dataframe(user_input))
        pool_encoded = self.preprocessor.transform(eligible_pool.drop(columns=['actual_price', 'fair_price', 'value_score']))
        
        similarities = cosine_similarity(user_encoded, pool_encoded).flatten()
        
        # 3. THE VALUE BOOST
        # Now, we find products that are:
        # (Prefered Brand/Stars) + (Best Value for Money)
        final_scores = (similarities * 0.7) + (eligible_pool['value_score'].values * 0.3)
        
        eligible_pool['recommendation_score'] = final_scores
        return eligible_pool.sort_values('recommendation_score', ascending=False).head(n).to_dict('records')
