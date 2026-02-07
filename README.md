# Crypto Trading Platform Churn Prediction

A machine learning project predicting user churn for crypto trading platforms, demonstrating end-to-end data science workflow from feature engineering to model deployment recommendations.

## 🎯 Project Overview

This project builds a predictive model to identify which crypto trading platform users are likely to become inactive (churn) within the next 30 days. The model combines simulated user behavior with real crypto market dynamics to create realistic predictions that could drive retention strategies.

**Key Skills Demonstrated:**
- Feature engineering for fintech/crypto products
- A/B testing metrics and experimental design
- Machine learning model development and comparison
- Model interpretability (SHAP values, feature importance)
- SQL-based scoring system for production deployment
- Business-focused insights and recommendations

## 📊 Business Context

Understanding churn drivers is critical for crypto platforms like Coinbase. This project answers:
- Which user behaviors predict churn?
- How do different user segments (spot traders vs. advanced traders) differ?
- What interventions could reduce churn?
- How would we A/B test retention strategies?

## 📁 Project Structure

```
crypto-churn-prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py                    # Main execution script
│   ├── data_generation.py         # Synthetic data creation
│   ├── feature_engineering.py     # Feature creation
│   ├── model_training.py          # ML model development
│   ├── model_evaluation.py        # Performance analysis
│   └── sql_queries.py             # Production SQL examples
├── exploratory_analysis.ipynb # EDA and visualizations
├── outputs/
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── shap_analysis.png
│   └── churn_segments.png
└── models/
    └── best_model.pkl
```

## 🔬 Methodology

### 1. Data Generation
- Simulated 50,000 users with realistic crypto trading patterns
- Incorporated real market volatility dynamics
- Created multiple user segments (casual, active, whale traders)

### 2. Feature Engineering
**Trading Behavior:**
- Trading frequency and recency
- Portfolio diversity (number of unique assets)
- Average trade size
- Response to market volatility

**Engagement Metrics:**
- Days since last login
- Feature adoption (yield products, advanced trading)
- Account age

**Market Context:**
- Recent market volatility exposure
- Portfolio performance

### 3. Model Development
Trained and compared:
- Logistic Regression (baseline, interpretable)
- Random Forest (ensemble method)
- XGBoost (gradient boosting, best performance)

### 4. Model Interpretability
- Feature importance rankings
- SHAP (SHapley Additive exPlanations) values for local and global interpretability
- Cohort analysis by user segment

## 📈 Key Findings

### Top Churn Predictors
1. **Days since last login** (strongest predictor)
2. **Trading frequency decline** (30-day trend)
3. **Portfolio diversity** (single-asset holders churn more)
4. **Yield product adoption** (non-adopters churn more)
5. **Market volatility response** (panic sellers churn)

### User Segmentation
- **High Risk:** Inactive 14+ days, low diversity, no yield adoption
- **Medium Risk:** Declining activity, moderate engagement
- **Low Risk:** Regular traders, diversified portfolios, multi-feature users

### Model Performance
- **XGBoost:** AUC-ROC 0.87, Precision@10% 0.73
- **Random Forest:** AUC-ROC 0.84, Precision@10% 0.68
- **Logistic Regression:** AUC-ROC 0.79, Precision@10% 0.61

### Proposed A/B Test
**Hypothesis:** Educational push notifications about yield products reduce churn by 15% for users inactive 7-14 days.

**Design:**
- Treatment: Daily educational content + yield product CTA
- Control: Standard re-engagement email
- Sample size: 10,000 users per group
- Duration: 30 days
- Primary metric: 30-day retention rate
- Secondary metrics: Yield adoption rate, trading reactivation

## 🛠️ Production Deployment

The `sql_queries.py` file contains production-ready SQL for:
- Daily churn risk scoring
- User segmentation
- Intervention targeting
- Performance monitoring

## 📚 Technical Details

**Libraries Used:**
- pandas, numpy (data manipulation)
- scikit-learn (modeling)
- xgboost (gradient boosting)
- shap (model interpretability)
- matplotlib, seaborn (visualization)
- sqlite3 (SQL examples)

**Model Selection Criteria:**
- AUC-ROC (discrimination)
- Precision@10% (actionable high-risk identification)
- Calibration (reliable probability estimates)
- Interpretability (business stakeholder communication)

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML workflow for a business problem
- Feature engineering for fintech/crypto domain
- Model comparison and selection
- Interpretation techniques beyond accuracy metrics
- Translation of model insights into business actions
- A/B test design for validating interventions
- SQL operationalization of ML models

