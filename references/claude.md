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
- Category
- Current Price
- Energy Rating
- Capacity
- Warranty
- Customer Rating
- Review Count
- Feature Specifications

Target Variable:
Price

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
