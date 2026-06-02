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
│   ├── 01_eda_ac.ipynb         # EDA for AC dataset
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

### AC Dataset — Current State (994 rows × 20 columns)

This is the only scraped dataset currently available. It is **clean with zero nulls**.

| Column | Type | Notes |
|---|---|---|
| product_name | object | Drop for modeling, keep for display |
| brand | object | 32 unique brands — encode carefully |
| price | float64 | **Target variable** |
| rating | float64 | r = -0.01 with price — **DROP** |
| capacity | float64 | r = 0.49 — **strongest predictor** |
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
| Voice Control | int64 | r = 0.17 — Merge into smart_features |
| PM 2.5 Filter | int64 | 10.7% presence — keep |
| 4-Way Air Swing | int64 | Check correlation before deciding |

### Features to KEEP (final list)
```
capacity, star_rating, inverter, brand, Self Diagnosis,
Hidden Panel Display, Auto Clean, Air Swing, PM 2.5 Filter,
4-Way Air Swing, smart_features (engineered), inverter_x_star (engineered)
```

### Features to DROP
```
product_name (display only), rating (r = -0.01), ac_type (89% imbalanced),
Dehumidification (87.5% constant), Turbo Mode (noise), Night Glow Buttons (zero signal)
```

### Features to ENGINEER
```python
# 1. Collapse Wi-Fi + APP Control + Voice Control into a smart tier score
df['smart_features'] = df['Wi-Fi Connectivity'] + df['APP Control'] + df['Voice Control']

# 2. Interaction term: inverter × star_rating captures premium tier
df['inverter_x_star'] = df['inverter'] * df['star_rating']
```

---

## ML PIPELINE — STAGE BY STAGE

### Stage 1 — EDA (notebook: 01_eda_ac.ipynb)

**Goal:** Understand the data before touching the model.

Plots to generate:
- Price distribution (histogram + log scale) — check for skew
- Price by brand (boxplot sorted by median) — brand premium visualization
- Correlation heatmap of all numeric features vs price
- Capacity vs price scatter with brand color coding
- smart_features distribution (bar chart of 0/1/2/3 counts)

**Gate to pass before moving on:**
- Confirm price distribution shape (normal? log-normal? bimodal?)
- Identify any obvious outliers (capacity = 0.75 → is this real or a scrape error?)
- Verify engineered features make intuitive sense

---

### Stage 2 — Preprocessing (src/preprocess.py)

**Goal:** Build a reusable, category-aware preprocessing function.

```python
def preprocess_ac(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop low-signal columns
    # 2. Engineer smart_features and inverter_x_star
    # 3. Encode brand (target encoding preferred over one-hot for tree models)
    # 4. Return clean feature matrix X and target vector y
```

**Brand encoding decision:**
- One-hot encoding → 32 new columns, interpretable, fine for tree models
- Target encoding → 1 column with mean price per brand, captures brand premium signal directly
- **Recommendation:** Try both in Stage 3. Target encoding usually wins with XGBoost.

**Capacity outlier check:**
- Rows with capacity < 1.0 (window ACs) — decide: drop them or keep and flag them?
- Make this a decision, not a silent assumption.

**Gate to pass:**
- Output shape should be ~994 × 12–14 columns
- Zero nulls after processing
- All dtypes should be float64 or int64 (no objects going into model)

---

### Stage 3 — Model Experiments (notebook: 03_model_experiments.ipynb)

**Goal:** Find the best model. Run all six, compare honestly.

**Evaluation protocol:**
```python
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)
# Metrics: RMSE, MAE, R² — report all three for each model
```

**Model progression:**
| # | Model | Why it's in the sequence |
|---|---|---|
| 1 | LinearRegression | Baseline — is price roughly linear? |
| 2 | Ridge | Same + L2 regularization |
| 3 | Lasso | L1 — automatic feature selection |
| 4 | DecisionTreeRegressor | Can it beat linear with no assumptions? |
| 5 | RandomForestRegressor | Ensemble — usually much better than single tree |
| 6 | XGBoostRegressor | Final target — boosting beats bagging on tabular |

**What "winning" means:**
- Primary criterion: R² on test set (5-fold CV mean)
- Secondary: RMSE — lower is better, but check if errors are clustered at high/low prices
- **Deployment threshold: R² ≥ 0.80** — if the best model doesn't clear this, do not build the Streamlit app yet. Debug features first.

**Log everything to a comparison table:**
```
| Model | CV R² mean | CV R² std | RMSE | MAE |
```

---

### Stage 4 — SHAP Explainability (notebook: 04_shap_explainability.ipynb)

**Goal:** Make the model's decisions human-readable.

```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Plots to generate:
# 1. shap.summary_plot() — global feature importance
# 2. shap.waterfall_plot() — single prediction explanation
# 3. shap.dependence_plot('capacity') — how capacity drives price
```

**What SHAP values will be used for in the app:**
- "Why is this priced at ₹X?" → waterfall plot for that product
- "What's driving the brand premium?" → brand SHAP contribution
- "What features should I compromise on to save money?" → feature impact on price reduction

**Gate:** SHAP values must sum (approximately) to prediction − base_value. If they don't, something is wrong in the explainer setup.

---

### Stage 5 — Prediction Function (src/predict.py)

```python
def predict_price(input_dict: dict, category: str = "ac") -> dict:
    """
    Returns:
    - predicted_price: float
    - shap_explanation: dict of feature -> contribution
    - alternatives: list of products within ±15% predicted price
    """
    model = joblib.load(f"models/{category}_xgboost_v1.pkl")
    # ... preprocess input_dict using category-specific preprocessor
    # ... run shap explainer
    # ... query processed dataset for alternatives
```

---

### Stage 6 — Streamlit App (app/streamlit_app.py)

**UI flow:**
1. User selects category (AC — only one for now, expandable)
2. User inputs product specs via dropdowns/sliders
3. App shows: predicted price band, "fair/overpriced/underpriced" verdict
4. SHAP waterfall chart: which features are driving the price up/down
5. Alternatives table: similar products sorted by value score

**Streamlit-specific rules:**
- Cache model loading with `@st.cache_resource`
- Cache data loading with `@st.cache_data`
- Use `st.columns()` for the main layout — predicted price on left, SHAP chart on right
- No `st.experimental_rerun()` — use session state properly

---

## MULTI-APPLIANCE EXPANSION (Future)

When a second appliance (washing machine, fridge, etc.) is scraped:

**Universal master schema (shared columns across all categories):**
```
product_name, brand, category, price, rating, review_count,
energy_rating, capacity_value, capacity_unit, warranty_years
```

**Capacity normalization problem:**
- AC → tons
- Fridge → litres
- WM → kg
- **These cannot go in the same column without normalization**
- Solution: `capacity_value` (float) + `capacity_unit` (string) as separate columns, OR normalize within-category to a percentile rank

**Appliance-specific columns:**
- Add sparse columns (e.g., `ac_inverter`, `wm_spin_speed`, `fridge_doors`)
- NaN for irrelevant categories — XGBoost handles this natively
- **Impute with 0 + add binary flag** if using linear models

**Model routing:**
```python
# One model per category, shared inference interface
model_registry = {
    "ac": "models/ac_xgboost_v1.pkl",
    "washing_machine": "models/wm_xgboost_v1.pkl",
}
```

---

## ERROR HANDLING SPEC

| Scenario | What to do |
|---|---|
| capacity listed in unexpected unit | Normalize in scraper. Log a warning if value seems off (e.g., capacity > 5 for AC = tons) |
| Product has no rating on site | Already handled — dataset has no nulls. For future scrapes, impute with category median |
| SHAP fails on a product | Fallback to feature importance from model.feature_importances_ |
| Model R² < 0.80 | Do not deploy. Go back to feature engineering. Log which features are hurting performance. |
| Capacity = 0.75 (window AC) | Investigate — if genuine, keep with a `is_window_ac` flag. If scrape error, drop. |

---

## TESTING CHECKLIST

Before calling any stage "done", verify:

**Data:**
- [ ] `df.isnull().sum()` → zero nulls in features after preprocessing
- [ ] `df.duplicated().sum()` → zero duplicates (or explicitly acknowledged)
- [ ] `df['price'].min()` > 0 — no zero-price rows
- [ ] capacity range is 0.75–3.0 — flag outliers

**Models:**
- [ ] Cross-validation uses `random_state=42` consistently
- [ ] Model artifacts saved to `models/` with version suffix
- [ ] All six models logged to comparison table before choosing winner

**App:**
- [ ] Streamlit runs locally with `streamlit run app/streamlit_app.py`
- [ ] Model loads from `models/` (not hardcoded inline)
- [ ] SHAP chart renders without error for at least 5 different inputs

---

## CONVERSATION PROTOCOL

**When Pradhuman shows code output, respond in this order:**
1. What the output means (interpret it, not just describe it)
2. Whether this is good, bad, or expected
3. What it implies about the next decision
4. One clear recommended next step (with reasoning)

**When Pradhuman asks "what should I do next?":**
- Check which stage he's on
- Check if the gate condition for that stage is met
- Give the next concrete, actionable task — not a vague direction

**When Pradhuman shares an error traceback:**
- Identify the root cause (not just the line number)
- Explain *why* this error occurs conceptually
- Give the minimal fix, not a full rewrite

**When Pradhuman shares model metrics:**
- Compare against the R² ≥ 0.80 deployment threshold
- Explain what RMSE/MAE mean in terms of rupees (e.g., "MAE of 3200 means the model is off by ₹3,200 on average")
- If metrics are poor, suggest which feature engineering step to try next

---

## PLACEMENT CONTEXT

This project should be framed in interviews as:
- "End-to-end ML pipeline on self-scraped real-world data"
- "Solved the schema unification problem for multi-appliance data"
- "Used SHAP for model interpretability to make predictions explainable to non-technical users"
- "Deployed as a Streamlit web app with a clean user-facing inference interface"

**Skills demonstrated:** Web scraping, data cleaning, feature engineering, regression modeling, ensemble methods (XGBoost), model explainability (SHAP), ML deployment (Streamlit), production pipeline thinking (versioned models, testing, error handling)

---

*Last updated: June 2026 | Project: Home Appliance Price Intelligence | Owner: Pradhuman, NIT Warangal*