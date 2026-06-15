"""
ApplianceRecommender — Upgraded
================================
Improvements over the original:
  1. Budget-aware hard filtering with a configurable tolerance band
  2. Fuzzy capacity matching (nearest-neighbor fallback when exact match is empty)
  3. Per-category feature relevance: similarity is computed only on columns
     that are meaningful for the queried category, preventing cross-category
     noise from poisoning the score.
  4. Weighted similarity components: brand, star_rating, inverter, smart
     features, and category-specific boolean features each get tunable weights.
  5. Diversity injection: ensures the top-n results don't all come from the
     same brand (unless the pool genuinely has no alternatives).
  6. Explainability dict: every returned product carries a human-readable
     breakdown of *why* it was recommended (match reasons, value verdict).
  7. Graceful degradation: capacity → budget → category — each constraint
     is relaxed in order when the pool is too thin.
  8. Category name normalisation: accepts 'AC', 'ac', 'Washing Machine',
     'WashingMachine', 'Refrigerator', 'fridge', etc.
  9. Validates user_input keys and raises clear errors early.
 10. All scores are logged so you can inspect them during development.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── Internal imports (adjust paths to your project layout) ──────────────────
from src.bi_layer import build_feature_dataframe, _load_model

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# Map the user-facing category names (case-insensitive, with common aliases)
# to the exact string that lives in X_train['category'].
_CATEGORY_ALIASES: Dict[str, str] = {
    "ac":               "AC",
    "air conditioner":  "AC",
    "airconditioner":   "AC",
    "split ac":         "AC",
    "window ac":        "AC",
    "refrigerator":     "Refrigerator",
    "fridge":           "Refrigerator",
    "ref":              "Refrigerator",
    "washing machine":  "WashingMachine",
    "washingmachine":   "WashingMachine",
    "washer":           "WashingMachine",
    "wm":               "WashingMachine",
}

# The capacity column that matters for each canonical category.
_CAPACITY_COL: Dict[str, str] = {
    "AC":             "capacity_ac_tons",
    "Refrigerator":   "capacity_ref_liters",
    "WashingMachine": "capacity_wm_kg",
}

# Feature groups used for weighted similarity.
# Only columns present in the dataset are used; missing ones are silently ignored.
_FEATURE_GROUPS: Dict[str, List[str]] = {
    "universal": [
        "rating", "n_features", "star_rating", "has_star_rating",
        "has_inverter", "has_wifi", "has_voice_control", "has_app_control",
        "smart_connectivity_score", "smart_intensity",
        "brand_frequency",                        # proxy for brand popularity
        "features_above_avg", "rating_above_avg", "capacity_above_avg",
    ],
    "AC": [
        "ac_split", "ac_window", "ac_pm25_filter", "ac_hepa_filter",
        "ac_auto_clean", "ac_hot_and_cold", "ac_copper_condenser",
        "ac_Dehumidification", "ac_Turbo Mode", "ac_Self Diagnosis",
        "ac_premium_features",
    ],
    "Refrigerator": [
        "ref_single_door", "ref_multi_door", "ref_chest_freezer",
        "ref_side_door", "ref_french_door", "ref_double_door",
        "ref_triple_door", "ref_frost_free", "ref_convertible",
        "ref_door_alarm", "ref_door_lock", "ref_dispenser",
        "ref_door_display", "ref_mini", "ref_door_complexity",
    ],
    "WashingMachine": [
        "wm_fully_automatic", "wm_semi_automatic", "wm_with_dryer",
        "wm_washer_only", "wm_dryer_only", "wm_front_load", "wm_top_load",
        "wm_inbuilt_heater", "wm_quick_wash", "wm_ss_tub",
        "wm_child_lock", "wm_shock_proof", "wm_display", "wm_tech_level",
    ],
}

# Weight assigned to each feature group during similarity computation.
# You can tune these to shift the recommender's personality.
_GROUP_WEIGHTS: Dict[str, float] = {
    "universal": 0.50,
    "category":  0.50,   # whichever category-specific group applies
}

# Weight split between similarity score and value score in the final ranking.
_SIMILARITY_WEIGHT: float = 0.65
_VALUE_WEIGHT:      float = 0.35

# Budget tolerance: if user sets a budget, we allow products up to this
# fraction above the budget (catches near-threshold products worth showing).
_BUDGET_TOLERANCE: float = 0.10   # 10 % above stated budget

# Capacity fuzzy tolerance: if no exact match, look for capacity within this
# relative tolerance of the requested value.
_CAPACITY_TOLERANCE: float = 0.15  # ± 15 %

# If one brand would dominate the top-n, cap it at this fraction.
_MAX_BRAND_FRACTION: float = 0.60  # e.g. max 2 out of 3 results from same brand


# ════════════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _normalise_category(raw: str) -> str:
    """Convert any user-facing category string to the canonical internal name."""
    key = raw.strip().lower()
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    # Already canonical?
    canonical_set = {"AC", "Refrigerator", "WashingMachine"}
    if raw.strip() in canonical_set:
        return raw.strip()
    raise ValueError(
        f"Unknown category '{raw}'. "
        f"Accepted values: {list(_CATEGORY_ALIASES.keys())}"
    )


def _select_feature_columns(category: str, available_cols: List[str]) -> List[str]:
    """
    Return the union of universal features and category-specific features,
    filtered to only columns that actually exist in the DataFrame.
    """
    wanted = set(_FEATURE_GROUPS["universal"]) | set(_FEATURE_GROUPS.get(category, []))
    return [c for c in available_cols if c in wanted]


def _weighted_similarity(
    user_vec: np.ndarray,
    pool_matrix: np.ndarray,
    user_cols: List[str],
    category: str,
) -> np.ndarray:
    """
    Compute a weighted cosine similarity that gives different importance to
    universal features vs category-specific features.

    Parameters
    ----------
    user_vec    : (1, n_features) array after preprocessor.transform
    pool_matrix : (n_products, n_features) array after preprocessor.transform
    user_cols   : column names corresponding to the feature axis
    category    : canonical category string

    Returns
    -------
    Weighted similarity array of shape (n_products,).
    """
    universal_mask = np.array([c in _FEATURE_GROUPS["universal"] for c in user_cols])
    category_mask  = np.array([c in _FEATURE_GROUPS.get(category, []) for c in user_cols])

    scores = np.zeros(pool_matrix.shape[0])
    total_weight = 0.0

    for mask, weight in [(universal_mask, _GROUP_WEIGHTS["universal"]),
                         (category_mask,  _GROUP_WEIGHTS["category"])]:
        if mask.sum() == 0:
            continue
        u_sub = user_vec[:, mask]
        p_sub = pool_matrix[:, mask]
        # Guard against zero-norm vectors (all-zeros row → cosine undefined)
        u_norm = np.linalg.norm(u_sub)
        if u_norm == 0:
            continue
        sim = cosine_similarity(u_sub, p_sub).flatten()
        scores += weight * sim
        total_weight += weight

    if total_weight > 0:
        scores /= total_weight   # normalise so scores stay in [0, 1]

    return scores


def _explain_recommendation(
    row: pd.Series,
    user_input: Dict[str, Any],
    category: str,
) -> Dict[str, str]:
    """
    Produce a human-readable dict explaining why this product was recommended.
    """
    reasons = []

    # Brand match
    user_brand = str(user_input.get("brand_name", "")).lower().strip()
    prod_brand = str(row.get("brand_name", "")).lower().strip()
    if user_brand and user_brand == prod_brand:
        reasons.append(f"Exact brand match ({prod_brand.title()})")
    elif not user_brand:
        reasons.append("No brand preference — showing best overall match")

    # Star rating
    user_stars = user_input.get("star_rating")
    prod_stars = row.get("star_rating", 0)
    if user_stars and prod_stars >= user_stars:
        reasons.append(f"{int(prod_stars)}-star rated (you wanted ≥{int(user_stars)}★)")
    elif prod_stars > 0:
        reasons.append(f"{int(prod_stars)}-star energy rating")

    # Inverter
    if user_input.get("has_inverter") and row.get("has_inverter", 0):
        reasons.append("Inverter technology ✓")

    # Smart features
    if row.get("smart_connectivity_score", 0) >= 2:
        smarts = []
        if row.get("has_wifi"):        smarts.append("WiFi")
        if row.get("has_app_control"): smarts.append("App control")
        if row.get("has_voice_control"): smarts.append("Voice control")
        if smarts:
            reasons.append("Smart: " + ", ".join(smarts))

    # Value verdict
    vs = row.get("value_score", 1.0)
    if vs >= 1.15:
        verdict = f"🔥 Great deal — fair price is ₹{row['fair_price']:,.0f} vs asking ₹{row['actual_price']:,.0f}"
    elif vs >= 1.05:
        verdict = f"✅ Good value — slightly underpriced vs market"
    elif vs >= 0.95:
        verdict = f"⚖️  Fair price — priced in line with market"
    else:
        verdict = f"💸 Premium priced — you're paying for brand/features"

    return {
        "match_reasons": "; ".join(reasons) if reasons else "Strong feature match",
        "value_verdict": verdict,
        "similarity_score": f"{row.get('similarity_score', 0):.3f}",
        "value_score":      f"{row.get('value_score', 1):.3f}",
        "final_score":      f"{row.get('recommendation_score', 0):.3f}",
    }


def _enforce_diversity(
    df: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """
    Prevent any single brand from taking more than _MAX_BRAND_FRACTION of
    the top-n slots.  Greedily picks the next best product that doesn't
    exceed the brand cap, then fills remaining slots without the cap.
    """
    max_per_brand = max(1, int(np.ceil(n * _MAX_BRAND_FRACTION)))
    selected_indices = []
    brand_counts: Dict[str, int] = {}

    for idx, row in df.iterrows():
        brand = str(row.get("brand_name", "unknown")).lower()
        if brand_counts.get(brand, 0) < max_per_brand:
            selected_indices.append(idx)
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        if len(selected_indices) == n:
            break

    # If we didn't fill all slots (very few brands in pool), top up without cap
    if len(selected_indices) < n:
        remaining = df.index.difference(selected_indices)
        selected_indices.extend(remaining[: n - len(selected_indices)])

    return df.loc[selected_indices]


# ════════════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ════════════════════════════════════════════════════════════════════════════

class ApplianceRecommender:
    """
    Production-grade recommender for Home Appliance Price Intelligence.

    Parameters
    ----------
    x_train_path : path to X_train.parquet
    y_train_path : path to y_train.parquet

    Usage
    -----
    recommender = ApplianceRecommender(
        x_train_path="models/dataset/X_train.parquet",
        y_train_path="models/dataset/Y_train.parquet",
    )

    results = recommender.get_recommendations(
        user_input={
            "category":       "AC",
            "capacity_value": 1.5,       # tons
            "budget":         45000,      # optional INR max budget
            "brand_name":     "lg",       # optional soft preference
            "star_rating":    3,          # optional min star rating
            "has_inverter":   1,          # optional
            "has_wifi":       0,          # optional
        },
        n=3,
    )
    """

    def __init__(
        self,
        x_train_path: str = "models/dataset/X_train.parquet",
        y_train_path: str = "models/dataset/Y_train.parquet",
    ):
        logger.info("Loading model and dataset …")
        self.pipeline, self.preprocessor, self.model, _ = _load_model()

        self.X_train = pd.read_parquet(Path(x_train_path))
        self.y_train = pd.read_parquet(Path(y_train_path))

        # Build the market pool once at startup
        self.pool = self.X_train.copy()
        self.pool["actual_price"] = np.expm1(self.y_train["log_price"])

        # Pre-compute fair prices and value scores for the entire pool
        log_preds = self.pipeline.predict(self.X_train)
        self.pool["fair_price"]  = np.expm1(log_preds)
        self.pool["value_score"] = self.pool["fair_price"] / self.pool["actual_price"]

        # Columns that the preprocessor expects (all X_train cols, no pool extras)
        self._model_cols = self.X_train.columns.tolist()

        logger.info(
            "Pool ready: %d products across %d categories.",
            len(self.pool),
            self.pool["category"].nunique(),
        )

    # ── PUBLIC API ───────────────────────────────────────────────────────────

    def get_recommendations(
        self,
        user_input: Dict[str, Any],
        n: int = 3,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-n recommended appliances for the given user preferences.

        Parameters
        ----------
        user_input : dict with the following keys:
            category       (str, required)  — e.g. 'AC', 'Refrigerator', 'Washing Machine'
            capacity_value (float, required) — tons / litres / kg depending on category
            budget         (float, optional) — max price in INR
            brand_name     (str, optional)  — preferred brand (soft constraint)
            star_rating    (int, optional)  — minimum BEE star rating (1–5)
            has_inverter   (int, optional)  — 1 to prefer inverter models
            has_wifi       (int, optional)  — 1 to prefer WiFi-enabled
            has_app_control(int, optional)  — 1 to prefer app-controlled
        n       : number of recommendations to return (default 3)
        verbose : if True, log intermediate pool sizes and scores

        Returns
        -------
        List of dicts, each containing full product features + scoring metadata
        + an 'explanation' dict (human-readable breakdown).
        """
        # ── 1. Validate & normalise inputs ──────────────────────────────────
        self._validate_input(user_input)
        category = _normalise_category(user_input["category"])
        capacity = float(user_input["capacity_value"])
        budget   = user_input.get("budget")
        cap_col  = _CAPACITY_COL[category]

        if verbose:
            logger.info("Category=%s | Capacity=%s | Budget=%s", category, capacity, budget)

        # ── 2. Hard filter: category ─────────────────────────────────────────
        pool = self.pool[self.pool["category"] == category].copy()

        if verbose:
            logger.info("After category filter: %d products", len(pool))

        if pool.empty:
            logger.warning("No products found for category '%s'.", category)
            return []

        # ── 3. Hard filter: budget (with tolerance band) ─────────────────────
        if budget is not None:
            budget_ceil = float(budget) * (1 + _BUDGET_TOLERANCE)
            within_budget = pool[pool["actual_price"] <= budget_ceil]
            if not within_budget.empty:
                pool = within_budget
                if verbose:
                    logger.info(
                        "After budget filter (≤₹%.0f + %.0f%% tolerance): %d products",
                        budget, _BUDGET_TOLERANCE * 100, len(pool),
                    )
            else:
                # Graceful degradation: no product fits budget → ignore budget constraint
                logger.warning(
                    "No products found within budget ₹%.0f (+%.0f%%). "
                    "Showing best available regardless of price.",
                    budget, _BUDGET_TOLERANCE * 100,
                )

        # ── 4. Hard filter: capacity (exact, then fuzzy fallback) ────────────
        pool, capacity_mode = self._filter_by_capacity(pool, cap_col, capacity, verbose)

        if pool.empty:
            logger.warning(
                "No products found for %s=%.2f even with fuzzy matching.",
                cap_col, capacity,
            )
            return []

        # ── 5. Weighted cosine similarity ────────────────────────────────────
        user_df  = build_feature_dataframe(user_input)
        user_enc = self.preprocessor.transform(user_df)           # (1, F)

        pool_raw = pool[self._model_cols].copy()
        pool_enc = self.preprocessor.transform(pool_raw)          # (N, F)

        feature_cols = _select_feature_columns(category, self._model_cols)
        # Map feature names to column indices in the encoded matrix.
        # NOTE: If your preprocessor produces a dense array aligned with
        # self._model_cols, the index mapping is straightforward. If it
        # re-orders columns, you need to track the output column order.
        # The implementation below works when preprocessor output columns
        # are in the same order as self._model_cols (standard for
        # ColumnTransformer with remainder='passthrough' or similar).
        col_index = {c: i for i, c in enumerate(self._model_cols)}
        selected_idx = [col_index[c] for c in feature_cols if c in col_index]

        if selected_idx:
            u_sub = user_enc[:, selected_idx]
            p_sub = pool_enc[:, selected_idx]
        else:
            u_sub = user_enc
            p_sub = pool_enc

        # Split into universal vs category-specific for weighted similarity
        universal_cols = [c for c in feature_cols if c in _FEATURE_GROUPS["universal"]]
        cat_cols       = [c for c in feature_cols if c in _FEATURE_GROUPS.get(category, [])]

        sim_scores = self._compute_weighted_similarity(
            user_enc, pool_enc, universal_cols, cat_cols, col_index
        )

        # ── 6. Final score = weighted(similarity + value_score) ──────────────
        value_scores = pool["value_score"].values.astype(float)
        # Clip extreme value scores to avoid outlier products dominating
        value_scores_clipped = np.clip(value_scores, 0.5, 2.0)
        # Normalise to [0, 1] range
        v_min, v_max = value_scores_clipped.min(), value_scores_clipped.max()
        if v_max > v_min:
            value_scores_norm = (value_scores_clipped - v_min) / (v_max - v_min)
        else:
            value_scores_norm = np.ones_like(value_scores_clipped) * 0.5

        final_scores = (
            _SIMILARITY_WEIGHT * sim_scores
            + _VALUE_WEIGHT     * value_scores_norm
        )

        pool = pool.copy()
        pool["similarity_score"]    = sim_scores
        pool["recommendation_score"] = final_scores

        if verbose:
            top_debug = pool[["brand_name", "actual_price",
                               "similarity_score", "value_score",
                               "recommendation_score"]].sort_values(
                "recommendation_score", ascending=False).head(10)
            logger.info("Top 10 before diversity:\n%s", top_debug.to_string())

        # ── 7. Sort and apply diversity ──────────────────────────────────────
        pool_sorted = pool.sort_values("recommendation_score", ascending=False)
        pool_diverse = _enforce_diversity(pool_sorted, n * 2)    # pre-select 2n, then pick n
        final_pool   = pool_diverse.head(n)

        # ── 8. Build output records ──────────────────────────────────────────
        output = []
        for _, row in final_pool.iterrows():
            record = row.to_dict()
            record["capacity_mode"] = capacity_mode   # 'exact' or 'fuzzy'
            record["explanation"]   = _explain_recommendation(row, user_input, category)
            output.append(record)

        return output

    # ── PRIVATE HELPERS ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_input(user_input: Dict[str, Any]) -> None:
        """Raise ValueError with a clear message if required keys are missing."""
        required = {"category", "capacity_value"}
        missing  = required - set(user_input.keys())
        if missing:
            raise ValueError(f"user_input is missing required keys: {missing}")
        if not isinstance(user_input["capacity_value"], (int, float)):
            raise TypeError(
                f"capacity_value must be numeric, got {type(user_input['capacity_value'])}"
            )

    @staticmethod
    def _filter_by_capacity(
        pool: pd.DataFrame,
        cap_col: str,
        capacity: float,
        verbose: bool,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Try exact match first; fall back to fuzzy (nearest available) if empty.

        Returns the filtered DataFrame and a mode string ('exact' or 'fuzzy').
        """
        exact = pool[pool[cap_col] == capacity]
        if not exact.empty:
            if verbose:
                logger.info("Exact capacity match (%s == %.2f): %d products", cap_col, capacity, len(exact))
            return exact, "exact"

        # Fuzzy: find products within _CAPACITY_TOLERANCE relative to requested value
        tol = capacity * _CAPACITY_TOLERANCE
        fuzzy = pool[
            (pool[cap_col] >= capacity - tol) &
            (pool[cap_col] <= capacity + tol)
        ]
        if not fuzzy.empty:
            nearest = pool[cap_col].sub(capacity).abs().min()
            logger.warning(
                "No exact capacity match for %s=%.2f. "
                "Using fuzzy match (±%.0f%%). Nearest available: %.2f.",
                cap_col, capacity, _CAPACITY_TOLERANCE * 100, capacity + nearest,
            )
            return fuzzy, "fuzzy"

        # Last resort: single nearest neighbour
        nearest_val = pool[cap_col].sub(capacity).abs().idxmin()
        nearest_pool = pool.loc[[nearest_val]]
        logger.warning(
            "No fuzzy match either. Returning single nearest product "
            "with %s=%.2f.", cap_col, pool.loc[nearest_val, cap_col]
        )
        return nearest_pool, "fuzzy"

    @staticmethod
    def _compute_weighted_similarity(
        user_enc:      np.ndarray,
        pool_enc:      np.ndarray,
        universal_cols: List[str],
        category_cols:  List[str],
        col_index:      Dict[str, int],
    ) -> np.ndarray:
        """
        Compute a two-component weighted cosine similarity.
        Each component uses only the relevant column subset of the
        preprocessor's output matrix.
        """
        scores       = np.zeros(pool_enc.shape[0])
        total_weight = 0.0

        for cols, weight in [
            (universal_cols, _GROUP_WEIGHTS["universal"]),
            (category_cols,  _GROUP_WEIGHTS["category"]),
        ]:
            idx = [col_index[c] for c in cols if c in col_index]
            if not idx:
                continue
            u_sub = user_enc[:, idx]
            p_sub = pool_enc[:, idx]
            u_norm = np.linalg.norm(u_sub)
            if u_norm == 0:
                continue
            sim     = cosine_similarity(u_sub, p_sub).flatten()
            scores += weight * sim
            total_weight += weight

        if total_weight > 0:
            scores /= total_weight

        return scores


# ════════════════════════════════════════════════════════════════════════════
#  PRETTY-PRINT UTILITY  (useful in Streamlit / notebooks)
# ════════════════════════════════════════════════════════════════════════════

def format_recommendation(rec: Dict[str, Any], rank: int) -> str:
    """Return a formatted string summary of one recommendation."""
    category = rec.get("category", "")
    cap_col  = _CAPACITY_COL.get(category, "")
    capacity = rec.get(cap_col, "?")

    lines = [
        f"{'='*60}",
        f"  #{rank}  {rec.get('brand_name', '').title()}  |  {category}  |  {capacity} {_capacity_unit(category)}",
        f"{'='*60}",
        f"  Price      : ₹{rec.get('actual_price', 0):,.0f}",
        f"  Fair Price : ₹{rec.get('fair_price', 0):,.0f}",
        f"  Rating     : {rec.get('rating', '?')}/5  ({rec.get('star_rating', 0):.0f}★ BEE)",
        f"  Inverter   : {'Yes' if rec.get('has_inverter') else 'No'}",
        f"  Smart Score: {rec.get('smart_connectivity_score', 0)}/3",
        f"",
        f"  🎯 Why recommended:",
        f"     {rec['explanation']['match_reasons']}",
        f"     {rec['explanation']['value_verdict']}",
        f"",
        f"  Scores → Similarity: {rec['explanation']['similarity_score']}  |  "
        f"Value: {rec['explanation']['value_score']}  |  "
        f"Final: {rec['explanation']['final_score']}",
        f"  Capacity match: {rec.get('capacity_mode', '?')}",
    ]
    return "\n".join(lines)


def _capacity_unit(category: str) -> str:
    return {"AC": "tons", "Refrigerator": "L", "WashingMachine": "kg"}.get(category, "")


# ════════════════════════════════════════════════════════════════════════════
#  QUICK TEST  (run: python recommender.py)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # Swap in your actual paths
    rec = ApplianceRecommender(
        x_train_path="models/dataset/X_train.parquet",
        y_train_path="models/dataset/y_train.parquet",
    )

    queries = [
        {
            "category":        "AC",
            "capacity_value":  1.5,
            "budget":          45000,
            "brand_name":      "lg",
            "star_rating":     3,
            "has_inverter":    1,
        },
        {
            "category":        "Washing Machine",
            "capacity_value":  7.0,
            "budget":          30000,
            "has_inverter":    0,
            "wm_fully_automatic": 1,
            "wm_front_load":   1,
        },
        {
            "category":        "Refrigerator",
            "capacity_value":  300,
            "budget":          25000,
            "brand_name":      "samsung",
            "star_rating":     2,
        },
    ]

    for q in queries:
        print(f"\n{'#'*60}")
        print(f"  Query: {q}")
        print(f"{'#'*60}")
        results = rec.get_recommendations(q, n=3, verbose=True)
        if not results:
            print("  ⚠  No recommendations found.")
        for rank, r in enumerate(results, start=1):
            print(format_recommendation(r, rank))