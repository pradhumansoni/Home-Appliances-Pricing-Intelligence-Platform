# Claude.md

# Project Name

AI-Powered Appliance Pricing Intelligence Platform

# Core Objective

Build a machine learning platform that helps consumers determine whether an appliance is fairly priced and discover better-value alternatives.

The machine learning model is not the final product.

The regression model is only one component of a larger pricing intelligence system.

# Business Problem

Consumers often see appliances on Amazon, Flipkart, Reliance Digital, Croma, and other marketplaces.

They frequently ask:

- Is this appliance worth the price?
- Is this product overpriced?
- Are there similar products offering better value?
- How much of the price comes from branding versus specifications?

This project answers those questions.

# User Flow

User enters:

- Product Name
- Brand
- Appliance Category
- Capacity
- Energy Rating
- Key Specifications
- Marketplace Price

System returns:

- Predicted Fair Price
- Premium/Discount Percentage
- Value Score
- Pricing Verdict
- Similar Alternatives

# Dataset

Primary Source:
Smartprix

Required Fields:

- Product Name
- Brand
- Appliance Category         # stored as `product` column; currently "AC", will expand to other appliances
- Model Year                 # stored as `model_year` column; ~16% missing in current scrape
- Current Price
- Energy Rating
- Capacity
- Warranty
- Customer Rating
- Review Count
- Feature Specifications

Target Variable:
Price

# Engineered Fields

Two columns are derived from the raw product `name` field during cleaning and have project-specific handling rules. Note: the `name` column is dropped from the final dataset, so these fields cannot be re-extracted without re-merging with raw scraped data.

## `product` (appliance category)
- Currently constant "AC" (only ACs were scraped so far)
- Purpose: enables the platform to scale to other appliances (refrigerators, washing machines, etc.) without restructuring
- **In training:** one-hot encode — treat as a categorical feature even though it has zero variance today; future-proofs the pipeline
- **In the app:** drives the appliance-category dropdown in the user flow
- Do NOT drop the column; constant value today is acceptable because the same pipeline will be reused for other appliances

## `model_year` (year of manufacture)
- Extracted via regex `\b(201[0-9]|202[0-9])\b` from the product name
- 16.4% of rows are missing — these need explicit handling
- **In EDA:** check year-vs-price relationship (expect newer = more expensive), and check whether missingness correlates with brand or price tier
- **Missing-value strategy (decided):**
  - Impute with **brand-level median year** (e.g., all missing Voltas rows → median year of known Voltas rows)
  - Add a binary `model_year_missing` indicator column so the model can learn whether "unknown year" itself carries signal
- **Feature transformation (decided):**
  - Transform to `age = current_year - model_year` — captures depreciation, stronger signal than raw year
  - `current_year` is determined at training time and again at inference time, so the transformation is consistent
- **In the inference API:** if the user does not provide model_year, treat as missing → apply the same brand-median imputation + missing-indicator logic from training
- **In the pricing intelligence engine:** age (or year) affects fair price — newer models of identical specs cost more, which is part of the "branding vs specs" decomposition in the explainability layer

# Machine Learning Roadmap

Phase 1:
Linear Regression

Phase 2:
Ridge Regression

Phase 3:
Lasso Regression

Phase 4:
Decision Tree Regressor

Phase 5:
Random Forest Regressor

Phase 6:
XGBoost Regressor

Model Selection:
Choose best model using:
- RMSE
- MAE
- R²

# Project Structure

data/
    raw/
    processed/

notebooks/
    eda.ipynb
    feature_engineering.ipynb
    model_training.ipynb

src/
    scraping/
    preprocessing/
    features/
    training/
    inference/

models/
    saved_models/

frontend/

backend/

# Development Workflow

1. Scrape Smartprix.
2. Store raw data.
3. Clean dataset.
4. Handle missing values.
5. Standardize specifications.
6. Perform EDA.
6.5. EDA specifically for `product` (note as constant) and `model_year` (year distribution, missingness pattern, year-vs-price scatter).
7. Create features.
8. Train baseline regression model.
9. Train advanced models.
10. Evaluate performance.
11. Save best model.
12. Build FastAPI service.
13. Build frontend dashboard.
14. Deploy application.

# Pricing Intelligence Engine

Input:
Product specifications + actual marketplace price

Prediction:
Fair market value

Analysis:

Difference = Actual Price - Predicted Fair Price

Generate:

- Fairly Priced
- Slightly Overpriced
- Significantly Overpriced
- Good Value

# Alternative Recommendation Engine

After pricing analysis:

1. Find appliances with similar specifications.
2. Rank by value-for-money.
3. Recommend lower-priced alternatives.

# Explainability

Use:

- SHAP
- Feature Importance

Questions answered:

- Why is this appliance expensive?
- Which features contributed most to the prediction?

# Resume Description

Built an AI-powered appliance pricing intelligence platform using Linear Regression, Ridge, Lasso, Random Forest, and XGBoost to estimate fair market value of consumer appliances, identify overpriced products, and recommend higher-value alternatives through interpretable machine learning.
