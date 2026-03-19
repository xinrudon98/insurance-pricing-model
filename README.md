# Insurance Pricing Regression Model

A regression-based modeling framework to evaluate insurance pricing adequacy and identify key drivers of premium rates using statistical analysis.

---

## Overview

This project develops statistical models to analyze insurance premium rates and assess pricing adequacy.

Using policy, geographic, and hazard-related features, the models identify key factors influencing premium variation and highlight segments with potential overpricing or underpricing.

---

## Architecture

Pricing Dataset  
↓  
Feature Engineering  
↓  
Regression Modeling (OLS / GLM)  
↓  
Model Evaluation  
↓  
Outlier Detection & Insights  

---

## Modeling Approach

### Feature Engineering
- Geographic variables (latitude, longitude)
- Hazard score segmentation
- Coverage limit banding (TIV / Coverage A)
- Monthly seasonality factors

### Model Development
- Ordinary Least Squares (OLS) regression
- Generalized Linear Model (GLM)
- Comparative modeling across different feature sets

### Scenario Analysis
- Models with and without seasonal variables
- Segment-based filtering for sensitivity testing
- Random-rate benchmarking for validation

---

## Key Features

### Multi-Model Framework
- Built multiple regression specifications to compare performance
- Evaluated impact of different feature groups on pricing behavior

### Outlier Detection
- Identified accounts outside expected pricing range using statistical thresholds
- Highlighted potential underwriting or pricing anomalies

### Pricing Adequacy Analysis
- Assessed whether premium rates align with expected risk drivers
- Provided insights into segments requiring adjustment

### Model Validation
- Compared actual vs. predicted premium rates
- Used random-rate models to validate explanatory power

---

## Repository Structure

insurance-pricing-model/
├── notebooks/
│   ├── primary_regression.ipynb
│   └── supplementary_regression.ipynb
├── data/
│   └── sample_pricing_data.xlsx
├── output/
│   └── .gitkeep
├── requirements.txt
└── README.md

---

## Tech Stack

- Python
- Pandas
- NumPy
- Statsmodels
- Matplotlib

---

## Setup

1. Place sample dataset in the `data/` folder  

2. Run the notebooks:
   - primary_regression.ipynb
   - supplementary_regression.ipynb

---

## Key Insights

- Geographic and hazard variables significantly influence premium rates  
- Coverage bands provide meaningful segmentation for pricing analysis  
- Seasonal effects can impact pricing patterns and should be evaluated separately  
- Outlier detection helps identify policies requiring underwriting review  

---

## Business Impact

- Improved understanding of pricing drivers  
- Identified potential pricing inefficiencies  
- Supported underwriting and pricing decision-making  
- Provided a structured approach to pricing validation  

---

## Future Improvements

- Incorporate additional risk features (claims history, exposure data)  
- Apply machine learning models (XGBoost, Random Forest)  
- Deploy model as an API for real-time pricing evaluation  
