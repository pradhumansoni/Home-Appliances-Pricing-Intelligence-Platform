# 🏠 Home Appliance Pricing Intelligence Platform

### AI-Powered Fair Price Prediction, Overpricing Detection & Smart Product Recommendations

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![SHAP](https://img.shields.io/badge/Explainable-AI-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

The Home Appliance Pricing Intelligence Platform is an end-to-end Machine Learning application that helps users determine whether a home appliance is fairly priced, overpriced, or underpriced based on its specifications.

The platform combines:

* Machine Learning
* Explainable AI (SHAP)
* Recommendation Systems
* Business Intelligence
* Interactive Streamlit Deployment

to provide data-driven purchasing insights for consumers.

---

## 🎯 Problem Statement

When purchasing appliances online, consumers often struggle to answer:

* Is this product fairly priced?
* Am I paying extra for the brand name?
* Are there better alternatives available?
* Which specifications contribute most to the price?

This project addresses these questions using a machine learning-based pricing intelligence system trained on real appliance data.

---

## 🚀 Key Features

### Fair Price Prediction

Predicts the expected market value of an appliance using its specifications.

Example:

| Product Price | Predicted Fair Price |
| ------------- | -------------------- |
| ₹52,999       | ₹46,200              |

---

### Overpricing Detection

Compares:

```text
Market Price
vs
Predicted Fair Price
```

and classifies products as:

* Underpriced
* Fairly Priced
* Slightly Overpriced
* Significantly Overpriced

---

### Smart Recommendation Engine

Suggests:

* Similar products
* Better-value alternatives
* Products with more features at similar prices

using feature similarity and pricing intelligence.

---

### Explainable AI (SHAP)

Every prediction is accompanied by model explanations showing:

* Features increasing price
* Features decreasing price
* Estimated brand premium
* Feature importance

---

### Business Intelligence Layer

Transforms model predictions into actionable insights such as:

* Product is 15% overpriced.
* Brand premium contributes approximately ₹6,000.
* Similar alternatives available within the same budget.

---

### Interactive Streamlit Dashboard

User-friendly interface supporting:

* Appliance specification inputs
* Fair price prediction
* Pricing verdicts
* SHAP explanations
* Alternative recommendations

---

## 📊 Dataset

The model is trained on a multi-category appliance dataset containing:

### Categories

* Air Conditioners
* Refrigerators
* Washing Machines

### Dataset Characteristics

```text
Rows: ~2900+
Features: 66+
Multiple Brands
Multiple Appliance Categories
```

---

## ⚙️ Machine Learning Pipeline

### 1. Data Collection

* Web Scraping
* Product Information Extraction
* Specification Parsing
* Category Standardization

---

### 2. Data Cleaning

* Missing Value Handling
* Duplicate Removal
* Data Type Corrections
* Category-specific preprocessing

---

### 3. Feature Engineering

Examples include:

* Smart Connectivity Score
* Feature Count
* Appliance Category Features
* Capacity Normalization
* Energy Efficiency Indicators
* Brand Encoding

---

### 4. Feature Encoding

* One-Hot Encoding
* Target Encoding
* Category Transformation

---

### 5. Model Training

Models evaluated:

| Model             |
| ----------------- |
| Linear Regression |
| Ridge Regression  |
| Lasso Regression  |
| Elastic Net       |
| Decision Tree     |
| Random Forest     |   
| Gradient Boosting |
| XGBoost           |

---

## 🏆 Final Model

### XGBoost Regressor

Performance:

```text
R² Score: 0.8872
```

Additional evaluation metrics:

* Cross Validation
* MAE
* RMSE
* Residual Analysis

The model is trained using log-transformed prices and converted back to the original scale during inference.

---

## 🧠 Explainable AI

The platform integrates SHAP (SHapley Additive exPlanations) to explain predictions.

Supported visualizations:

* Global Feature Importance
* Local Prediction Explanations
* Waterfall Plots
* Feature Impact Analysis

This enables transparent and trustworthy predictions.

---

## 🔍 Recommendation Engine

The recommendation system uses:

* Cosine Similarity
* Feature Matching
* Predicted Fair Prices
* Market Prices

to identify appliances offering better value.

### Outputs

* Similar Products
* Better Alternatives
* Value-for-Money Suggestions
* Price Advantage Analysis

---

## 🏗️ System Architecture

```text
User Input
    │
    ▼
Preprocessing Pipeline
    │
    ▼
XGBoost Model
    │
    ▼
Fair Price Prediction
    │
    ├── SHAP Explainability
    │
    ├── Business Intelligence Layer
    │
    └── Recommendation Engine
    │
    ▼
Streamlit Dashboard
```

---

## 🛠️ Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* XGBoost

### Explainability

* SHAP

### Visualization

* Matplotlib
* Plotly

### Deployment

* Streamlit

### Model Serialization

* Joblib

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/pradhumansoni/AI-Powered-Appliance-Pricing-Intelligence-Platform.git
```

Move into the project directory:

```bash
cd AI-Powered-Appliance-Pricing-Intelligence-Platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Example Use Cases

### Consumers

* Check if a product is overpriced.
* Compare competing products.
* Understand pricing drivers.

### Retailers

* Pricing strategy analysis.
* Product benchmarking.
* Competitive intelligence.

### Manufacturers

* Brand premium estimation.
* Feature pricing analysis.
* Product positioning insights.

---

## 🎓 Skills Demonstrated

### Data Science

* Exploratory Data Analysis
* Feature Engineering
* Regression Modeling
* Model Evaluation
* Hyperparameter Tuning

### Machine Learning

* Ensemble Learning
* XGBoost
* Pipeline Construction
* Cross Validation

### Explainable AI

* SHAP
* Model Interpretability
* Feature Attribution

### Recommendation Systems

* Similarity Search
* Cosine Similarity
* Ranking Systems

### Software Engineering

* Modular Architecture
* Production Pipelines
* Model Serialization
* Streamlit Deployment

---

## 🔮 Future Improvements

* Real-time price tracking
* Automated scraping pipelines
* LLM-powered product insights
* Personalized recommendations
* Price trend forecasting
* Retailer comparison engine

---

## 👨‍💻 Author

### Pradhuman Soni

Student at NIT Warangal, 
Persuing M.Sc. Mathematics & Scientific Computing

---



### If you found this project useful, consider giving it a ⭐ on GitHub.
