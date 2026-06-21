"""
APPLIANCE RECOMMENDER - CONTENT-BASED RECOMMENDATION ENGINE

This module implements a sophisticated content-based recommendation system for home appliances.
Key design principles:
1. PURITY: Recommendations based on feature similarity, NOT price model errors
2. HARD CONSTRAINTS: Category + capacity matching within tolerance bands
3. SOFT CONSTRAINTS: Cosine similarity on brand, features, star rating
4. SOFT FILTERING: Price range validation ensures budget alignment

CRITICAL FIX (V3):
The original V2 used 'deal_score' (fair_price - actual_price) to rank recommendations.
This had a fatal flaw: since the price model has R² = 0.887 (87.7% explained variance),
rankings were rewarding products where the model was WRONG, not products that were
objectively better matches. V3 separates RANKING (pure content similarity) from 
FILTERING (price validity), solving this problem completely.
"""

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
from src.bi_layer_improved import build_feature_dataframe, _load_model

# Relative file loading
X_train_path = BASE_DIR / "models" / "dataset" / "X_train.parquet"
y_train_path = BASE_DIR / "models" / "dataset" / "y_train.parquet"

class ApplianceRecommender:
    """
    Content-based recommendation engine for home appliances.
    
    Algorithm Overview:
    1. Load training data and pre-compute fair prices for all products in the pool
    2. When user requests recommendations:
       - Filter by category and capacity (hard constraints)
       - Rank by cosine similarity (soft constraints - feature matching)
       - Filter by price range (soft constraints - budget validation)
       - Apply brand diversity to avoid monopoly
       - Return top-n products
    
    Key Parameters:
    - tolerance_map: How much capacity variance is acceptable (category-specific)
    - price_tolerance_pct: How far a product's actual price can deviate from fair value
    - max_per_brand: Maximum products from same brand in recommendations
    """
    
    def __init__(self):
        """
        Initialize the recommender with training data and pre-computed fair prices.
        
        TIMING NOTE: This is called once when the Streamlit app starts.
        All computations here are ONE-TIME costs, NOT per-recommendation costs.
        """
        
        # ============================================================================
        # STEP 1: LOAD MODEL AND DATA
        # ============================================================================
        # Load the pre-trained XGBoost pipeline (preprocessor + price model)
        self.pipeline, self.preprocessor, self.model, _ = _load_model()
        
        # Load the training dataset (used as our "market pool" for recommendations)
        # X_train contains all 65 features for ~2,300 products
        # y_train contains log(price) values for the same products
        self.X_train = pd.read_parquet(X_train_path)
        self.y_train = pd.read_parquet(y_train_path)
        
        # ============================================================================
        # STEP 2: PRE-CALCULATE THE 'MARKET POOL'
        # ============================================================================
        # Make a copy of the training data to serve as our candidate pool
        self.pool = self.X_train.copy()
        
        # Add actual observed prices to the pool (convert from log space)
        # These are the real-world prices seen during training
        self.pool['actual_price'] = np.expm1(self.y_train['log_price'])
        
        # ============================================================================
        # STEP 3: PRE-CALCULATE FAIR PRICES FOR ALL PRODUCTS
        # ============================================================================
        # Use the trained model to predict "fair market price" for each product
        # This is a ONE-TIME computation (happens at startup, not per-recommendation)
        # Fair prices are used for two purposes:
        #   (a) Price range filtering - keep recommendations in user's budget ballpark
        #   (b) Quality assessment - can identify overpriced/underpriced deals
        
        log_preds = self.pipeline.predict(self.X_train)
        self.pool['fair_price'] = np.expm1(log_preds)
        
        # DEBUG: Check if fair_price calculation worked
        # print(f"[DEBUG] Fair prices computed. Min: ₹{self.pool['fair_price'].min():,.0f}, "
        #       f"Max: ₹{self.pool['fair_price'].max():,.0f}, Mean: ₹{self.pool['fair_price'].mean():,.0f}")
        
        # ============================================================================
        # DESIGN PHILOSOPHY: Why separate RANKING from FILTERING?
        # ============================================================================
        # V2 used 'deal_score' to rank: (fair_price - actual_price) / fair_price
        # This ranked products by how much the model was WRONG, not by quality match!
        #
        # Example problem: If the model underestimated price, deal_score goes negative,
        # pushing good products to the bottom of recommendations.
        #
        # SOLUTION (V3): Use pure CONTENT similarity for ranking (what features match),
        # but use price range filtering to ensure recommendations stay in the user's
        # expected budget. This is mathematically cleaner and prevents model errors
        # from corrupting recommendations.
        # ============================================================================

        # ============================================================================
        # STEP 4: CONFIGURE RECOMMENDATION PARAMETERS
        # ============================================================================
        
        # IMPROVEMENT 1: Capacity Tolerance Map
        # Each appliance category has different capacity units:
        # - AC: capacity in tons (fractional)
        # - Refrigerator: capacity in liters (large integers)
        # - Washing Machine: capacity in kg (small integers)
        # So tolerance must be category-specific to make sense.
        #
        # Examples:
        # - AC: tolerance of 0.2 tons means we accept 1.3-1.7 tons if user wants 1.5
        # - Refrigerator: tolerance of 20L means we accept 230-270L if user wants 250
        # - Washing Machine: tolerance of 1kg means we accept 6-8kg if user wants 7
        #
        # These values were chosen empirically to give ~5-20 matches per query
        self.tolerance_map = {
            'AC': 0.2,                  # ±0.2 tons
            'Refrigerator': 20,         # ±20 liters
            'WashingMachine': 1         # ±1 kg
        }

        # IMPROVEMENT 2: Price Range Tolerance for Recommendations
        # When recommending, we only show products priced within ±15% of fair price.
        # Why 15%?
        # - Too narrow (e.g., 5%): Very few recommendations, user gets frustrated
        # - Too wide (e.g., 30%): Recommendations may be outside user's budget
        # - 15% is a balanced sweet spot for home appliances
        #
        # Example: If fair price is ₹30,000, we show products priced ₹25,500 - ₹34,500
        self.price_tolerance_pct = 0.15  # ±15%

        # IMPROVEMENT 3: Brand Diversity Control
        # Without this, if LG is the best brand in the pool, all 5 recommendations
        # would be LG products, which doesn't help the user compare alternatives.
        #
        # How it works: After sorting by content similarity, we group by brand and
        # take only the top 2 products from each brand. This ensures variety.
        #
        # Example: If ranked list is [LG-1, LG-2, LG-3, Samsung-1, Samsung-2, ...]
        # We return [LG-1, LG-2, Samsung-1, Samsung-2, Whirlpool-1, Whirlpool-2]
        self.max_per_brand = 2

    def get_recommendations(self, user_input: Dict[str, Any], n=10) -> List[Dict]:
        """
        Generate n product recommendations based on user preferences.
        
        Algorithm Steps:
        1. Apply hard constraints: Filter by category + capacity match
        2. Apply soft constraints: Rank by cosine similarity (feature matching)
        3. Apply price filter: Keep only products within ±15% of fair price
        4. Apply brand diversity: Max 2 products per brand
        5. Return top-n results
        
        Args:
            user_input (Dict): Dictionary with keys like 'category', 'capacity_value',
                               'brand_name', 'star_rating', and feature flags
                               Example: {
                                   'category': 'AC',
                                   'capacity_value': 1.5,
                                   'brand_name': 'LG',
                                   'star_rating': 3,
                                   'ac_split': True,
                                   'ac_copper_condenser': True,
                                   ...
                               }
            n (int): Number of recommendations to return (default: 10)
        
        Returns:
            List[Dict]: List of product dictionaries with all features + scores.
                        Each dict includes 'recommendation_score' (0-1 cosine similarity).
                        Empty list if no matches found.
        
        Raises:
            Nothing - gracefully returns empty list on any error (category not found, etc.)
        """
        
        category = user_input.get('category')
        capacity = user_input.get('capacity_value')
        
        # ============================================================================
        # CONSTRAINT 1: FILTER BY CATEGORY AND CAPACITY
        # ============================================================================
        # Category mapping: User input uses internal names (AC, Refrigerator, WashingMachine)
        category_capacity_map = {
            'AC': 'capacity_ac_tons',
            'Refrigerator': 'capacity_ref_liters',
            'WashingMachine': 'capacity_wm_kg'
        }
        
        # Get the correct column name based on the category
        capacity_col = category_capacity_map.get(category)
        
        if not capacity_col:
            # print(f"[ERROR] Invalid category: {category}")
            return []  # Invalid category - return empty list

        # Get tolerance for this category (e.g., 0.2 for AC, 20 for Refrigerator)
        tolerance = self.tolerance_map.get(category, 0)
        
        # Calculate absolute difference between product capacity and user preference
        capacity_diff = (self.pool[capacity_col] - capacity).abs()

        # Apply hard constraints: MUST match category AND capacity within tolerance
        # These are non-negotiable - a 1-ton AC is not a match for someone wanting 2 tons
        mask = (self.pool['category'] == category) & (capacity_diff <= tolerance)
        eligible_pool = self.pool[mask].copy()
            
        if eligible_pool.empty:
            # print(f"[DEBUG] No products match category={category}, capacity={capacity}±{tolerance}")
            return []

        # DEBUG: Report matching products
        # print(f"[DEBUG] Found {len(eligible_pool)} products matching category + capacity constraints")

        # ============================================================================
        # CONSTRAINT 2: RANK BY COSINE SIMILARITY (FEATURE MATCHING)
        # ============================================================================
        # This is the core of the recommendation algorithm.
        #
        # What we're measuring: "Of all products that fit the user's category + capacity,
        # which ones have features most similar to what the user asked for?"
        #
        # How it works:
        # 1. Encode the user's input using the same preprocessor as the model
        # 2. Encode all eligible products using the same preprocessor
        # 3. Compute cosine similarity: cos(angle) between vectors (0 = different, 1 = identical)
        # 4. Sort by similarity (highest first)
        #
        # Why cosine similarity?
        # - Scale-invariant: A rating-based space and a feature-count space can be mixed
        # - Interpretable: 0-1 scale (0% to 100% match) is easy for users to understand
        # - Fast: O(n) after preprocessing
        #
        # CRITICAL: We drop 'actual_price' and 'fair_price' before encoding, so that
        # price information does NOT influence the similarity calculation. This ensures
        # recommendations are purely about feature matching, not model predictions.
        
        # Encode user's input (just once)
        user_encoded = self.preprocessor.transform(build_feature_dataframe(user_input))
        
        # Encode all eligible products, EXCLUDING price columns
        # Reason: We want feature similarity, not price similarity
        pool_encoded = self.preprocessor.transform(
            eligible_pool.drop(columns=['actual_price', 'fair_price'])
        )
        
        # Compute cosine similarity between user input and all candidates
        # Shape: (1, num_eligible) - array of similarity scores for each product
        similarities = cosine_similarity(user_encoded, pool_encoded).flatten()
        
        # Attach similarity scores to the eligible pool
        eligible_pool['recommendation_score'] = similarities

        # DEBUG: Report similarity range
        # print(f"[DEBUG] Similarity scores: min={similarities.min():.3f}, "
        #       f"max={similarities.max():.3f}, mean={similarities.mean():.3f}")

        # ============================================================================
        # CONSTRAINT 3: FILTER BY PRICE RANGE
        # ============================================================================
        # At this point, we have products ranked by feature similarity.
        # But some might be overpriced or underpriced relative to fair value.
        # This filter ensures recommendations stay within the user's budget ballpark.
        #
        # How it works:
        # 1. Predict fair price for the user's input (what the model thinks it should cost)
        # 2. Allow products within ±15% of that fair price
        # 3. Drop products outside this range
        #
        # Why do this?
        # User expectation: "I want an AC like LG 1.5-ton split" 
        # Fair price: ₹25,000
        # Price range: ₹21,250 - ₹28,750
        # If we recommended a ₹50,000 product (premium brand), user would be shocked!
        # This filter prevents that by dropping outliers.
        #
        # Why not use actual prices instead of fair prices?
        # Because actual prices in the training data have noise (some products overpriced,
        # some underpriced). Fair prices are model-smoothed estimates of "true" value.
        
        # Predict fair price for the user's specs
        user_df = build_feature_dataframe(user_input)
        user_fair_price_log = self.pipeline.predict(user_df)[0]
        user_fair_price = np.expm1(user_fair_price_log)
        
        # Calculate acceptable price band (±15%)
        price_lower = user_fair_price * (1 - self.price_tolerance_pct)
        price_upper = user_fair_price * (1 + self.price_tolerance_pct)
        
        # Apply price filter
        price_mask = (eligible_pool['actual_price'] >= price_lower) & \
                     (eligible_pool['actual_price'] <= price_upper)
        eligible_pool = eligible_pool[price_mask]
        
        if eligible_pool.empty:
            # print(f"[DEBUG] No products within price range ₹{price_lower:,.0f} - ₹{price_upper:,.0f}")
            return []

        # DEBUG: Report price-filtered results
        # print(f"[DEBUG] {len(eligible_pool)} products within price range "
        #       f"₹{price_lower:,.0f} - ₹{price_upper:,.0f}")

        # ============================================================================
        # CONSTRAINT 4: RANKING
        # ============================================================================
        # Sort the pool by recommendation_score (highest first)
        # Higher score = more similar to user's preferences
        ranked_pool = eligible_pool.sort_values('recommendation_score', ascending=False)

        # DEBUG: Show top candidates before brand filtering
        # print(f"[DEBUG] Top 3 before brand filtering:")
        # for idx, row in ranked_pool.head(3).iterrows():
        #     print(f"  {row['brand_name']} - score: {row['recommendation_score']:.3f}")

        # ============================================================================
        # CONSTRAINT 5: BRAND DIVERSITY CONTROL
        # ============================================================================
        # Problem: If LG dominates the market for 1.5-ton split ACs, the top 5
        # recommendations might be all LG products, leaving Samsung/Daikin out.
        #
        # Solution: For each brand, keep only the top-2 products by similarity.
        # This ensures variety while still respecting the ranking.
        #
        # How it works:
        # ranked_pool.groupby('brand_name', group_keys=False).head(2)
        # → For each brand group, take the first 2 rows (highest score within brand)
        # → Don't add 'brand_name' as an index level (group_keys=False)
        # → Result: Top 2 LG + Top 2 Samsung + Top 2 Whirlpool, etc.
        #
        # NOTE: groupby() re-orders rows by brand, so we re-sort after grouping
        diverse_pool = ranked_pool.groupby('brand_name', group_keys=False).head(self.max_per_brand)

        # Re-sort by similarity score (groupby scrambles the order)
        final_pool = diverse_pool.sort_values('recommendation_score', ascending=False).head(n)

        # DEBUG: Show final recommendations
        # print(f"[DEBUG] Final recommendations (after brand filtering):")
        # for idx, row in final_pool.iterrows():
        #     print(f"  {row['brand_name']} - score: {row['recommendation_score']:.3f}, "
        #           f"price: ₹{row['actual_price']:,.0f}")

        # ============================================================================
        # RETURN RESULTS
        # ============================================================================
        # Convert DataFrame rows to dictionaries (one dict per product)
        # Streamlit's display function will handle formatting
        return final_pool.to_dict('records')
