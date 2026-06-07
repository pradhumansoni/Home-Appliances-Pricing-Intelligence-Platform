# CLAUDE.md — Home Appliance Price Intelligence (Smartprix)

## Streamlit ML App | Python | Pradhuman | NIT Warangal Placement Portfolio

---

## ⚠️ CRITICAL OPERATING INSTRUCTIONS — READ THIS FIRST

**Your role in this project is an assistant, explainer, and navigator — NOT the primary coder.**

- **Pradhuman writes most of the code himself.** Do not auto-generate full scripts or pipelines unless explicitly asked.
- **Your primary job is:**
  1. Explaining what the output of his code means (errors, metrics, plots, model scores)
  2. Telling him what the next step is and *why*
  3. Answering "why is this happening?" and "what should I do next?"
  4. Helping him debug when he's stuck, with targeted fixes — not full rewrites
  5. Reviewing code he writes and giving precise, educational feedback
- **When Pradhuman shows you output** (a DataFrame, a metric, a plot, an error traceback), your job is to interpret it, explain it, and tell him what decision to make.
- **Do not silently proceed to the next stage.** After each stage, stop and summarize: what was accomplished, what the numbers mean, and what the options are going forward.
- **Do not overwrite or delete files** without being explicitly asked.
- **Do not make architectural decisions unilaterally.** Present options with tradeoffs, let Pradhuman decide.
- **Ask before writing code** that would replace something he already built.

---

## PROJECT OVERVIEW

**What this project does:**
A Streamlit web app that answers three questions for any home appliance listed on Smartprix:
1. Is this product fairly priced for what it offers?
2. Am I paying a brand premium?
3. Are there better-value alternatives at a similar price?

**Why this project matters (placement framing):**
- End-to-end ML pipeline: scraping → preprocessing → feature engineering → modeling → explainability → deployment
- Real-world data, not a Kaggle CSV
- Demonstrates XGBoost, SHAP, Streamlit, and production-grade pipeline thinking
- Solves a problem real users care about

---

## PROJECT STRUCTURE

```
appliance-price-intelligence/
│
├── data/
│   ├── raw/                    # Never touch these — original scraped files
│   │   └── ac_cleaned.csv      # 994 rows, 20 columns (already done)
│   ├── processed/              # Output of preprocessing scripts
│   │   └── ac_features.csv
│   └── master/                 # Unified multi-appliance dataset (future)
│       └── appliances_master.csv
│
├── notebooks/
│   ├── 01_eda_ac.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_shap_explainability.ipynb
│
├── src/
│   ├── preprocess.py           # Cleaning + feature engineering per category
│   ├── train.py                # Model training + evaluation
│   ├── predict.py              # Inference + routing function
│   └── utils.py                # Shared helpers
│
├── models/
│   └── ac_xgboost_v1.pkl       # Versioned saved models
│
├── app/
│   └── streamlit_app.py        # Streamlit UI
│
├── tests/
│   └── test_preprocess.py      # At minimum: test schema, dtypes, no nulls
│
├── CLAUDE.md                   # This file
└── requirements.txt
```

---

## TECH STACK

| Purpose | Library |
|---|---|
| Data manipulation | pandas, numpy |
| Visualization (notebooks) | matplotlib, seaborn, plotly |
| Preprocessing | scikit-learn (Pipeline, ColumnTransformer, OrdinalEncoder, OneHotEncoder, SimpleImputer) |
| ML models | scikit-learn, xgboost |
| Explainability | shap |
| Serialization | joblib |
| App deployment | streamlit |
| Validation (optional) | pydantic |

---

## DATA SCHEMA

### AC Dataset — Current State (994 rows × 21 columns)

This is the only scraped dataset currently available. It has zero nulls after cleaning.

| Column | Type | Notes |
|---|---|---|
| product_name | object | Drop for modeling, keep for display |
| brand | object | 32 unique brands — encode carefully |
| price | float64 | **Target variable** |
| rating | float64 | r = -0.01 with price — **DROP** |
| capacity | float64 | r = 0.49 — strongest predictor |
| star_rating | int64 | Energy efficiency stars — keep |
| ac_type | object | 89.4% Split — **DROP** |
| inverter | int64 | Binary — keep |
| Dehumidification | int64 | 87.5% = 1, near constant — **DROP** |
| Turbo Mode | int64 | r ≈ 0.01 — **DROP** |
| Self Diagnosis | int64 | Balanced split — keep |
| Hidden Panel Display | int64 | r = 0.10 — keep |
| Night Glow Buttons | int64 | r ≈ 0.00 — **DROP** |
| Auto Clean | int64 | Reasonable signal — keep |
| Air Swing | int64 | r = 0.10 — keep |
| Wi-Fi Connectivity | int64 | Merge into smart_features |
| APP Control | int64 | Merge into smart_features |
| Voice Control | int64 | r = 0.17 — merge into smart_features |
| PM 2.5 Filter | int64 | 10.7% presence — keep |
| 4-Way Air Swing | int64 | Check correlation before deciding |
| model_year | int64 | Extracted via regex from product_name; ~155 rows imputed with group-median (brand + capacity bucket) + year_is_imputed flag |

### Features to KEEP
```
capacity, star_rating, inverter, brand, Self Diagnosis, Hidden Panel Display,
Auto Clean, Air Swing, PM 2.5 Filter, 4-Way Air Swing,
smart_features (engineered), inverter_x_star (engineered),
model_year, year_is_imputed
```

### Features to DROP
```
product_name (display only), rating (r = -0.01), ac_type (89% imbalanced),
Dehumidification (87.5% constant), Turbo Mode (noise), Night Glow Buttons (zero signal)
```

### Features to ENGINEER
```python
# 1. Smart tier score — collapse three smart features into one ordinal signal
df['smart_features'] = df['Wi-Fi Connectivity'] + df['APP Control'] + df['Voice Control']
# Values: 0 to 3

# 2. Premium smart flag — 1 if all three smart features are present
df['premium_smart_ac'] = (
    (df['Wi-Fi Connectivity'] == 1) &
    (df['APP Control'] == 1) &
    (df['Voice Control'] == 1)
).astype(int)

# 3. Inverter × star interaction — captures the premium tier effect
df['inverter_x_star'] = df['inverter'] * df['star_rating']

# 4. Capacity × star interaction — higher-tonnage 5-star ACs are disproportionately expensive
df['capacity_star_interaction'] = df['capacity'] * df['star_rating']

# 5. Feature richness score — count of all premium features, proxy for product tier
df['feature_richness_score'] = (
    df['Wi-Fi Connectivity'] + df['APP Control'] + df['Voice Control'] +
    df['inverter'] + df['Auto Clean'] + df['Self Diagnosis']
)
# Values: 0 to 6

# 6. Brand premium flag — group brands into premium vs standard based on EDA
# Validate this grouping from data (price by brand boxplot) — don't assume
df['premium_brand'] = df['brand'].isin(['Daikin', 'Mitsubishi', ...]).astype(int)

# 7. model_year — extracted via regex from product_name
# ~155 rows where regex fails: impute with group-median (brand + capacity bucket)
# Always create the flag column alongside the imputed column
df['year_is_imputed'] = 0  # set to 1 where imputed
```

**Feature addition rule:** Add a feature only if it improves CV R² by at least 0.005 AND has a clear business reason. Don't add noise.

---

## ML PIPELINE — STAGE BY STAGE

### Stage 1 — Dataset Audit + EDA (notebook: 03.ac_eda.ipynb) ✅ **COMPLETE**

**Goal:** Understand the data fully before writing a single line of preprocessing code.

#### 1a. Dataset Audit ✅

Run this before any plots or decisions:

- **Shape report:** number of rows, number of columns, data types breakdown (numerical / binary / categorical)
- **Missing values table:** missing count + missing percentage per column
- **Duplicate scan:** exact duplicates (`df.duplicated().sum()`) + near-duplicates check
- **Target variable deep analysis:**
  - Stats: mean, median, standard deviation, skewness, kurtosis
  - Plots: histogram, KDE plot, boxplot of `price`
  - Is the distribution normal? Log-normal? Bimodal? This affects modeling decisions.

#### 1b. Univariate Analysis ✅

For every numerical feature: histogram + boxplot.
For every categorical feature: countplot.

Questions to answer:
- Most common brands?
- Most common capacities?
- Most common star ratings?

#### 1c. Bivariate Analysis ✅

- Scatterplots: each numerical feature vs price
- Boxplots: price grouped by each categorical/binary feature
- Key questions: Which brands command premium prices? Does Wi-Fi increase price? Does inverter increase price?

#### 1d. Correlation Analysis ✅

- Pearson correlation matrix of all numerical features
- Heatmap — document all feature-feature and feature-target correlations above 0.80
- Capacity vs price scatter with brand color coding
- smart_features distribution bar chart (counts of 0/1/2/3)

**Gate to pass before moving to Stage 2:**
- [x] Price distribution shape confirmed — right-skewed, log-normal; `log1p` normalizes it. Use for linear models.
- [x] Capacity outlier question resolved — original cleaning rounded 24 distinct values to 9 standard tiers, creating a phantom 0.75T cluster of 14 rows. Reverted to continuous capacity. Only 1 row sits at exactly 0.75T (MarQ Split AC, ₹19,990) — kept as-is.
- [x] Engineered features verified to make intuitive sense — defer to Stage 2 (will be validated by the baseline linear model in the recursive workflow).
- [x] Model year missingness pattern analyzed — bimodal by brand (10 brands at 0%, several at 30%+, iMee and Toshiba at 100% with n=1 each), uniform by price tier (~0.16 across quartiles). The `year_is_imputed` indicator is **dropped** (no signal).
- [x] Smart-features multicollinearity confirmed — Wi-Fi Connectivity ↔ APP Control at r ≈ 0.99; Voice ↔ both at r ≈ 0.80. Collapsing into `smart_features = Voice + Wi-Fi + App` is well-justified.
- [x] Brand premium signal confirmed — median-sorted brand boxplot shows a clear premium ladder (O General, Panasonic, Haier at top; Sansui, MarQ, Sharp at bottom). Brand encoding will be critical.
- [x] Capacity correlation with price: r = **0.50** (after revert to continuous values; was 0.49 with rounded tiers).

---

### Stage 2 — Data Cleaning + Preprocessing (src/preprocess.py)

**Goal:** Build a reusable, category-aware preprocessing function. Every decision made here must be explicit and documented.

```python
def preprocess_ac(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop low-signal columns
    # 2. Handle outliers (document every removal)
    # 3. Impute missing values per strategy below
    # 4. Engineer all features from the Features to ENGINEER section
    # 5. Encode brand
    # 6. Return clean feature matrix X and target vector y
```

#### Missing Value Strategy

- Numerical columns → median imputation
- Categorical columns → mode imputation
- `model_year` specifically → group-median imputation by (brand + capacity bucket), with `year_is_imputed` flag set to 1 for affected rows

#### Outlier Handling Policy

- **Do NOT remove genuine premium products.** A ₹1.5L Daikin is not an outlier — it is data.
- Remove only: data-entry errors and impossible values (e.g., capacity = 0, negative price).
- Document the reason for every removal in a comment. If you can't justify it, don't remove it.

#### Brand Encoding Decision

- One-hot encoding → 32 new columns, interpretable, fine for tree models
- Target encoding → 1 column with mean price per brand, captures brand premium signal directly
- **Recommendation:** Try both in Stage 3. Target encoding usually wins with XGBoost on this kind of high-cardinality categorical. Fit target encoding on train set only — compute mean price per brand from X_train, then apply to X_test using that mapping (never re-fit on test).

#### Capacity Outlier Check

- Rows with capacity < 1.0 (window ACs) — decide: drop or keep with `is_window_ac` flag?
- Make this an explicit decision. Document it here once resolved.

**Gate to pass:**
- Output shape: ~994 × 14–16 columns
- Zero nulls after processing
- All dtypes are float64 or int64 — no object columns entering the model

---

### Stage 3 — Train-Test Split

Split immediately after preprocessing, before any encoding decisions or model training.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

- **80% Train / 20% Test**, `random_state=42` for reproducibility
- **CRITICAL — Data leakage rule:** Never touch `X_test` during feature selection, encoding decisions, or hyperparameter tuning. Any decision informed by test data = contaminated results. `X_test` is touched exactly once: final evaluation of the chosen model.
- All cross-validation and hyperparameter tuning runs on `X_train` only.

---

### Stage 4 — Model Experiments (notebook: 03_model_experiments.ipynb)

**Goal:** Find the best model. Run all models, compare honestly, don't skip ahead to XGBoost.

#### Evaluation Protocol

```python
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Report: RMSE, MAE, R² for every model
```

#### Model Progression

| # | Model | Purpose |
|---|---|---|
| 1 | LinearRegression | Baseline — is price roughly linear in features? |
| 2 | Ridge | L2 regularization — handles multicollinearity |
| 3 | Lasso | L1 regularization — automatic feature selection |
| 4 | ElasticNet | Combines Ridge + Lasso — good when unsure which penalty fits |
| 5 | DecisionTreeRegressor | Nonlinear baseline — no linearity assumption |
| 6 | RandomForestRegressor | Bagging ensemble — usually beats single tree significantly |
| 7 | XGBoostRegressor | Primary target — boosting beats bagging on tabular data |

#### Per-Model Tuning Notes

**Ridge**
- Tune: `alpha` — controls L2 penalty strength
- Log the alpha value that gives best CV R²

**Lasso**
- Tune: `alpha`
- After tuning, extract and document which features were zeroed out — this is analytical insight, not just a number

**ElasticNet**
- Tune: `alpha` + `l1_ratio`
- Use when Lasso removes too many features but Ridge doesn't remove enough

**Decision Tree**
- Tune: `max_depth`, `min_samples_leaf`, `min_samples_split`
- Expect overfitting without tuning — this is a learning opportunity

**Random Forest**
- Tune: `n_estimators`, `max_depth`, `min_samples_leaf`

**XGBoost**
- Tune: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
- Use early stopping on a held-out validation set during initial tuning to avoid over-fitting to CV

#### Hyperparameter Tuning Strategy

Run tuning only after confirming the untuned baseline beats R² = 0.70. Tuning a fundamentally broken model is wasted effort.

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}

search = RandomizedSearchCV(
    xgb_model, param_dist, n_iter=50,
    cv=5, scoring='r2', random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)
```

After RandomizedSearchCV identifies a promising region, use GridSearchCV to zoom in on the best hyperparameter combination. Log best params alongside final scores.

#### Model Comparison Table

Log all results here before picking a winner:

```
| Model            | CV R² mean | CV R² std | Test RMSE | Test MAE |
|------------------|------------|-----------|-----------|----------|
| LinearRegression |            |           |           |          |
| Ridge            |            |           |           |          |
| Lasso            |            |           |           |          |
| ElasticNet       |            |           |           |          |
| DecisionTree     |            |           |           |          |
| RandomForest     |            |           |           |          |
| XGBoost          |            |           |           |          |
```

**Deployment threshold: R² ≥ 0.80.** If the best model does not clear this, do not build the Streamlit app. Go back to feature engineering — log which features the model finds uninformative and investigate why.

**What metrics mean in rupees:** Always translate RMSE and MAE into ₹ terms when reviewing — "MAE of 3200 means the model is off by ₹3,200 on average". Also check if errors are clustered at high or low prices (a residuals vs fitted plot will show this).

---

### Stage 5 — SHAP Explainability (notebook: 04_shap_explainability.ipynb)

**Goal:** Make the winning model's decisions human-readable — for the app and for your own understanding of what the model learned.

```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Required plots:
# 1. shap.summary_plot()          — global feature importance ranked by mean |SHAP|
# 2. shap.waterfall_plot()        — single prediction explanation (pick 3–5 products)
# 3. shap.dependence_plot('capacity')  — how capacity drives price across the range
```

**What SHAP values will be used for in the Streamlit app:**
- "Why is this AC priced at ₹X?" → waterfall plot rendered for that product
- "What is the brand premium costing me?" → brand SHAP contribution isolated
- "What feature should I drop to save money?" → bottom features in waterfall

**Gate:** SHAP values for each prediction must sum approximately to `prediction − base_value`. If they don't, the explainer setup is wrong — check that the model and preprocessor are aligned.

---

### Stage 6 — Prediction Function (src/predict.py)

```python
def predict_price(input_dict: dict, category: str = "ac") -> dict:
    """
    Returns:
        predicted_price:   float — point estimate
        confidence_range:  tuple — (lower_bound, upper_bound) based on model error distribution
        shap_explanation:  dict  — feature -> SHAP contribution
        alternatives:      list  — products within ±15% of predicted price, sorted by value score
    """
    model = joblib.load(f"models/{category}_xgboost_v1.pkl")
    # 1. Preprocess input_dict using the category-specific preprocessor
    # 2. Run prediction
    # 3. Run SHAP explainer on the input
    # 4. Query processed dataset for alternatives within price band
```

---

### Stage 7 — Streamlit App (app/streamlit_app.py)

**UI flow:**
1. User selects appliance category (AC — only one for now, expandable)
2. User inputs product specs via dropdowns and sliders
3. App returns: predicted price band + "fair / overpriced / underpriced" verdict
4. SHAP waterfall chart: which features push the price up or down for this specific product
5. Alternatives table: similar products sorted by value score (features per rupee)

**Streamlit-specific rules:**
- Cache model loading with `@st.cache_resource`
- Cache data loading with `@st.cache_data`
- Use `st.columns()` for the main layout — predicted price on the left, SHAP chart on the right
- No `st.experimental_rerun()` — use session state properly

---

## MULTI-APPLIANCE EXPANSION (Future)

When a second appliance category (washing machine, fridge, etc.) is scraped:

**Universal master schema — shared columns across all categories:**
```
product_name, brand, category, price, rating, review_count,
energy_rating, capacity_value, capacity_unit, warranty_years
```

**Capacity normalization problem:**
- AC → tons
- Fridge → litres
- Washing machine → kg
- These cannot go in the same column without normalization.
- Solution: `capacity_value` (float) + `capacity_unit` (string) as separate columns, or normalize within-category to a percentile rank before combining.

**Appliance-specific sparse columns:**
- Add category-specific columns (e.g., `ac_inverter`, `wm_spin_speed`, `fridge_doors`)
- NaN for categories where the column is irrelevant — XGBoost handles NaN natively
- For linear models: impute with 0 + add a binary `has_<feature>` flag

**Model routing:**
```python
# One model per category, shared inference interface
model_registry = {
    "ac":              "models/ac_xgboost_v1.pkl",
    "washing_machine": "models/wm_xgboost_v1.pkl",
}
```

---

## ERROR HANDLING SPEC

| Scenario | What to do |
|---|---|
| Capacity in unexpected unit | Normalize in scraper. Log a warning if value is implausible (e.g., capacity > 5 for an AC in tons). |
| Product has no rating on site | Already handled — current dataset has no nulls. For future scrapes, impute with category median. |
| model_year regex fails on a product | Set `year_is_imputed = 1`. Impute with group-median by (brand + capacity bucket). |
| SHAP fails on a product | Fallback to `model.feature_importances_` for a simplified explanation. |
| Model R² < 0.80 | Do not deploy. Return to feature engineering. Log which features the model weights low and investigate. |
| Capacity = 0.75 (window AC) | Investigate first. If genuine: keep and add `is_window_ac = 1` flag. If scrape error: drop and document. |

---

## TESTING CHECKLIST

Before calling any stage "done":

**Data:**
- [x] `df.isnull().sum()` — only `model_year` has 16.4% missing; all other columns zero nulls.
- [x] `df.duplicated().sum()` — zero duplicates (cleaning step removed them).
- [x] `df['price'].min()` > 0 — min is ₹19,990, no zero-price rows.
- [x] `df['capacity']` range is 0.6–3.0 (after revert from rounded tiers) — 24 distinct values, all within plausible AC range.
- [x] `model_year` is present, range 2018–2026. **`year_is_imputed` indicator DROPPED** per EDA decision (uniform missingness across price = no signal).

**Models:**
- [ ] Cross-validation uses `random_state=42` consistently across all models
- [ ] Model artifacts saved to `models/` with version suffix
- [ ] All seven models logged to the comparison table before choosing a winner
- [ ] Ridge, Lasso, and ElasticNet — alpha values (and l1_ratio for ElasticNet) logged alongside R² scores
- [ ] Lasso zeroed-out feature list documented
- [ ] Best hyperparameters for XGBoost logged

**App:**
- [ ] Streamlit runs locally with `streamlit run app/streamlit_app.py`
- [ ] Model loads from `models/` — not hardcoded inline
- [ ] SHAP chart renders without error for at least 5 different input combinations

---

## CONVERSATION PROTOCOL

**When Pradhuman shows code output, respond in this order:**
1. What the output means — interpret it, don't just describe it
2. Whether this is good, bad, or expected
3. What it implies about the next decision
4. One clear recommended next step with reasoning

**When Pradhuman asks "what should I do next?":**
- Identify which stage he is currently in
- Verify the gate condition for that stage is met
- Give the next concrete, actionable task — not a vague direction

**When Pradhuman shares an error traceback:**
- Identify the root cause, not just the line number
- Explain *why* this error occurs conceptually
- Give the minimal fix — not a full rewrite

**When Pradhuman shares model metrics:**
- Compare against the R² ≥ 0.80 deployment threshold explicitly
- Translate RMSE and MAE into rupees
- If metrics are poor, suggest the specific feature engineering step most likely to fix it

---

## PLACEMENT CONTEXT

This project should be framed in interviews as:
- "End-to-end ML pipeline on self-scraped real-world data"
- "Solved the schema unification problem for multi-appliance data"
- "Used SHAP for model interpretability — predictions are explainable to non-technical users"
- "Deployed as a Streamlit web app with a clean user-facing inference interface"

**Skills demonstrated:** Web scraping, data cleaning, feature engineering, regression modeling (linear through ensemble), model explainability (SHAP), hyperparameter optimization, ML deployment (Streamlit), production pipeline thinking (versioned models, testing, error handling)

---

*Last updated: June 2026 | Project: Home Appliance Price Intelligence | Owner: Pradhuman, NIT Warangal*