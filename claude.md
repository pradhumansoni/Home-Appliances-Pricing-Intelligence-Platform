# CLAUDE.md — Home Appliance Price Intelligence (Smartprix)

## Streamlit ML App | Python | Pradhuman | NIT Warangal Placement Portfolio

---

# PURPOSE OF THIS FILE

This document is the **Project Encyclopedia**.

It preserves:

* Historical decisions,
* Experimental findings,
* EDA observations,
* Modeling rationale,
* Feature engineering decisions,
* Interview framing,
* Technical trade-offs.

This file answers:

> "Why does the project look the way it does?"

---

# SOURCE OF TRUTH

Execution order is governed by:

> **AGENTS.md**

This document provides context and rationale.

If conflicts arise:

> **AGENTS.md takes precedence for execution.**

---

# ⚠️ CRITICAL OPERATING INSTRUCTIONS

Your role is:

> Assistant, explainer, reviewer, and navigator.

NOT the primary coder.

Pradhuman writes production code.

---

## Primary Responsibilities

1. Explain outputs.
2. Explain errors.
3. Recommend next steps.
4. Review code.
5. Discuss trade-offs.
6. Interpret results.
7. Help debug.

---

## Avoid

Do NOT:

* Rewrite working systems.
* Skip validation steps.
* Make architectural pivots unilaterally.
* Ignore historical decisions.
* Generate full implementations without being asked.

---

# PROJECT OVERVIEW

This project evolved from a price prediction notebook into an:

> **AI-Powered Appliance Pricing Intelligence Platform**

The objective is to help consumers answer:

## Estimation Questions

* What is the fair price of this appliance?
* Why is this the estimated price?
* What similar products offer good value?

## Pricing Intelligence Questions

* Is the observed market price justified?
* Am I paying a brand premium?
* How much money could I save?
* Which alternatives provide better value?

---

# PROJECT EVOLUTION

The project evolved through several stages.

## Phase 1

AC-only regression.

Goal:

Predict AC prices.

---

## Phase 2

Investigate model limitations.

Discovery:

AC-only data imposed a structural ceiling.

---

## Phase 3

Multi-category expansion.

Added:

* Refrigerators,
* Washing Machines,
* Microwaves.

Goal:

Increase data diversity.

Improve generalization.

---

## Phase 4

Transition from:

> Prediction System

to

> Pricing Intelligence Platform.

Focus shifted toward:

* Consumer usefulness,
* Explainability,
* Decision support.

---

# HISTORICAL RECORD

The sections below describe completed work.

These phases are complete.

They should NOT be revisited unless explicitly instructed.

---

# COMPLETED PHASES

## AC EDA

Completed.

Key findings:

* Price distribution right-skewed.
* log1p target beneficial.
* Capacity strongest predictor.
* Brand premium signal exists.
* Smart features exhibit multicollinearity.

---

## Feature Engineering

Completed.

Important decisions:

Kept:

* smart_features
* inverter × star
* feature_count
* age

Rejected:

* premium_brand
* is_window_ac

Linear R²:

0.471 → 0.483

---

## AC Model Experiments

Completed.

Nine models benchmarked.

Best:

XGBoost One-Hot

R²:

0.524

Key finding:

Data ceiling identified.

---

## Multi-Category Expansion

Completed.

Categories:

* AC
* Refrigerator
* Washing Machine
* Microwave

Completed:

* Parsing
* Cleaning
* EDA
* Modeling
* Evaluation
* SHAP

---

# HISTORICAL LESSONS

## Data Ceiling

AC-only performance plateaued.

Reason:

Missing market variables.

Not model failure.

---

## Outliers

Premium products are legitimate observations.

Do NOT remove them without evidence.

---

## Explainability

SHAP is mandatory.

Predictions without explanations reduce business usefulness.

---

# CURRENT STAGE

The project has moved beyond model development.

Current focus:

> Business Intelligence Layer.

Model experimentation is complete.

The objective is now productization.

---

# BUSINESS INTELLIGENCE PHILOSOPHY

The BI Layer sits above the ML model.

It:

* Does NOT modify predictions.
* Interprets predictions.
* Converts outputs into decisions.

The ML model answers:

> What is the fair price?

The BI Layer answers:

> What should the user do?

---

# DUAL USER JOURNEYS

## Estimation Mode

Used when:

Observed market price is unknown.

Questions answered:

* Fair price?
* Why?
* Similar alternatives?

---

## Pricing Intelligence Mode

Used when:

Observed market price exists.

Questions answered:

* Fair or overpriced?
* Savings?
* Brand premium?
* Better alternatives?

---

# PARTIAL INPUT PHILOSOPHY

Consumers rarely know every specification.

The product should remain useful under incomplete information.

The system should support:

> Graceful degradation.

Missing information should be interpreted as:

> Unknown.

NOT invalid.

---

# MINIMUM INPUT PRINCIPLE

Predictions must work using:

* Category,
* Brand,
* Capacity Value,
* Capacity Unit.

Everything else should remain optional.

Prediction quality should improve as additional information becomes available.

---

# IMPORTANT IMPLEMENTATION RULE

Do NOT redesign the model solely for partial inputs.

Reuse existing models.

Support incomplete inputs through:

* Validation,
* Preprocessing,
* Inference logic.

Avoid maintaining multiple models.

---

## Updates for Business Intelligence Layer Completion

### New Section: Business Intelligence Implementation

**Completed on**: ['14-06-2026']

#### Core Components Implemented:

1. **Fair Price Estimation System**
   - Handles both complete and partial feature inputs
   - Returns:
     - Predicted fair price (₹)
     - Prediction range (confidence interval)
     - SHAP explanation of top features

2. **Pricing Verdict Engine**
   ```python
   | Difference   | Verdict             |
   | ------------ | ------------------- |
   | ±10%         | Fairly Priced       |
   | +10–20%      | Slightly Overpriced |
   | >20%         | Overpriced          |
   | −10% to −20% | Good Deal           |
  <−20%        | Exceptional Value   |
   ```

3. **Brand Premium Analysis**
   - Quantifies brand contribution using SHAP values
   - Example output:  
     "About ₹2,500 of the estimated price is attributable to brand effects."

4. **Partial Input Handling**
   - Minimum required inputs:
     - Category
     - Brand
     - Capacity Value
     - Capacity Unit
   - Graceful degradation for missing features
   - Automatic default value imputation

#### Technical Implementation Details:

```python
# Key Functions
def get_fair_price(features: Dict) -> Dict:
    """Main prediction endpoint"""
    pass

def get_pricing_verdict(observed, predicted) -> Dict:
    """Generate pricing assessment"""
    pass

def interpret_brand_premium(shap_values) -> str:
    """Explain brand contribution"""
    pass
```

#### Validation Testing:

1. **Test Coverage**:
   - 100+ test cases across categories
   - Edge cases (missing brand, extreme values)
   - Cross-category consistency

2. **Performance**:
   - Prediction latency: <500ms
   - Memory usage: <1GB

### Updated Project Status

```markdown
## CURRENT PROJECT STATUS

### COMPLETED PHASES

[Previous completed phases...]

NEW: ✅ Business Intelligence Layer
- Fair price estimation
- Pricing verdict system  
- Brand premium analysis
- Partial input support

### UPCOMING PHASES

⏳ Recommendation Engine Integration
⏳ Streamlit Application
⏳ Deployment

# RECOMMENDATION PHILOSOPHY

Recommendations exist to support decisions.

They are NOT an isolated recommender system.

Recommendations should prioritize:

1. Similarity,
2. Savings,
3. Value.

---

# INTERVIEW FRAMING

Describe this project as:

> "An AI-powered appliance pricing intelligence platform built on self-scraped real-world data."

NOT:

> "A regression model."

Emphasize:

* End-to-end ownership,
* Explainability,
* Product thinking,
* Decision support,
* Consumer usefulness.

---

# HONEST LIMITATIONS

AC-only:

R² ≈ 0.52

Reason:

Structural ceiling.

---

Multi-category:

Expected improvement:

0.70–0.78.

---

Automatic retraining:

Deferred.

Reason:

Out of scope.

The project focuses on:

> ML engineering and product development.

NOT full MLOps.

---

# FUTURE ENHANCEMENTS

Potential additions:

* Drift monitoring,
* Scheduled retraining,
* Weighted recommendations,
* User preference personalization,
* Recommendation feedback loops.

These are optional.

Do NOT prioritize them over the current roadmap.

---

# CONVERSATION PROTOCOL

When interpreting outputs:

1. Explain what happened.
2. Explain whether it is expected.
3. Explain implications.
4. Recommend the next step.

---

# PLACEMENT OBJECTIVE

The ultimate objective is not merely deployment.

The objective is to produce a flagship project demonstrating:

* Real-world data acquisition,
* Feature engineering,
* Regression modeling,
* Explainability,
* Business intelligence,
* Product design,
* Practical ML engineering.

This project should be memorable, defensible, and interview-ready.
