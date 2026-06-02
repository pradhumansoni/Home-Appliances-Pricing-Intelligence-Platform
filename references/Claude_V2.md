# CLAUDE.md V2

## PROJECT NAME
AI-Powered Appliance Pricing Intelligence Platform

## PRIMARY GOAL
Build a production-grade ML platform that estimates fair market value for appliances and helps users identify overpriced products and better-value alternatives.

## BUSINESS PROBLEM
Consumers struggle to determine whether appliance prices are justified. Brands often charge premiums, and users lack objective tools to evaluate value.

## SUCCESS CRITERIA
- Predict fair market value accurately.
- Explain prediction drivers.
- Calculate value-for-money score.
- Recommend better alternatives.
- Deploy a complete web application.

## TECH STACK
Backend:
- Python
- FastAPI
- PostgreSQL

ML:
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP

Frontend:
- Next.js
- Tailwind

Deployment:
- Vercel
- Render

## DATA PIPELINE
1. Scrape Smartprix.
2. Store raw data.
3. Clean and validate.
4. Engineer features.
5. Train models.
6. Evaluate models.
7. Save best model.
8. Serve through API.

## MODEL ROADMAP
Stage 1:
Linear Regression

Stage 2:
Ridge Regression

Stage 3:
Lasso Regression

Stage 4:
Decision Tree

Stage 5:
Random Forest

Stage 6:
XGBoost

## EVALUATION
Use:
- RMSE
- MAE
- R²
- Cross Validation

## USER WORKFLOW
User enters:
- Product specs
- Marketplace price

System returns:
- Fair value estimate
- Premium percentage
- Value score
- Verdict
- Similar alternatives

## API DESIGN
POST /predict
POST /analyze
POST /alternatives
GET /health

## VALUE SCORE LOGIC
Compare:
Actual Price vs Predicted Fair Value

Generate:
Excellent Value
Good Value
Fairly Priced
Overpriced

## EXPLAINABILITY
Use SHAP.
Display top positive and negative contributors.

## MILESTONES

Milestone 1:
Scraper complete.

Milestone 2:
Dataset cleaned.

Milestone 3:
EDA completed.

Milestone 4:
Linear Regression baseline.

Milestone 5:
Advanced models trained.

Milestone 6:
Best model selected.

Milestone 7:
API completed.

Milestone 8:
Frontend completed.

Milestone 9:
Deployment completed.

## RESUME DESCRIPTION
Built an AI-powered appliance pricing intelligence platform using Linear Regression, Ridge, Lasso, Random Forest, and XGBoost to estimate fair market value, analyze pricing efficiency, explain model decisions using SHAP, and recommend higher-value alternatives through a full-stack web application.
