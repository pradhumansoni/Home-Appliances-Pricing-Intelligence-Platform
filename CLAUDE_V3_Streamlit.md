
# CLAUDE.md V3 (STREAMLIT EDITION)

## PROJECT NAME
AI-Powered Appliance Pricing Intelligence Platform

## OBJECTIVE
Build a full machine learning application using only Python and Streamlit.

The system should help users answer:

- Is this appliance fairly priced?
- Am I paying a brand premium?
- Are there better-value alternatives?

The regression model is only one subsystem.

---

## TECH STACK

Python 3.11+

Data Collection:
- requests
- beautifulsoup4

Data Processing:
- pandas
- numpy

Machine Learning:
- scikit-learn
- xgboost

Explainability:
- shap

Visualization:
- matplotlib
- plotly

Deployment:
- streamlit

Model Persistence:
- joblib

---

## PROJECT STRUCTURE

appliance-pricing-platform/

data/
  raw/
  processed/

notebooks/
  01_scraping.ipynb
  02_eda.ipynb
  03_feature_engineering.ipynb
  04_model_training.ipynb

src/
  scraper.py
  preprocess.py
  train.py
  predict.py
  recommend.py
  explain.py

models/
  best_model.pkl
  preprocessor.pkl

app.py

README.md

CLAUDE.md

---

## DATA REQUIREMENTS

Source:
Smartprix

Collect:

- Product Name
- Brand
- Category
- Price
- Rating
- Review Count
- Energy Rating
- Capacity
- Warranty
- Specifications

Target:
Price

---

## MACHINE LEARNING ROADMAP

Stage 1
Linear Regression

Stage 2
Ridge Regression

Stage 3
Lasso Regression

Stage 4
Decision Tree

Stage 5
Random Forest

Stage 6
XGBoost

Metrics:

- RMSE
- MAE
- R²

---

## PRICING INTELLIGENCE ENGINE

User Inputs:

- Brand
- Category
- Capacity
- Energy Rating
- Warranty
- Actual Marketplace Price

System Predicts:

- Fair Market Value

Calculates:

- Premium Percentage
- Discount Percentage
- Value Score

Returns:

- Excellent Value
- Good Value
- Fairly Priced
- Overpriced

---

## RECOMMENDATION ENGINE

Find products with:

- Similar category
- Similar capacity
- Similar energy rating

Rank by:

- Value Score
- Customer Rating

Return Top 5 alternatives.

---

## SHAP EXPLAINABILITY

Show:

Positive price drivers

Negative price drivers

Example:

+ Premium Brand
+ Higher Capacity
+ Better Energy Rating

---

## STREAMLIT PAGES

Page 1:
Appliance Analyzer

Page 2:
Alternative Recommendations

Page 3:
Model Explanations

---

## DEVELOPMENT PRIORITIES

P0:
Data Pipeline
EDA
Linear Regression

P1:
Advanced Regression Models

P2:
Pricing Engine
Recommendations
SHAP

P3:
Streamlit Deployment

---

## RESUME DESCRIPTION

Built an AI-powered appliance pricing intelligence platform using Linear Regression, Ridge, Lasso, Random Forest, and XGBoost to estimate fair market value, identify pricing premiums, explain predictions with SHAP, and recommend better-value alternatives through a Streamlit application.
