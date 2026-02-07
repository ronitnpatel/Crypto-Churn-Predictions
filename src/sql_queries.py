"""
SQL Queries Module
Production-ready SQL for churn prediction deployment
"""

# SQL Query Templates for Production Deployment

DAILY_CHURN_SCORING = """
-- Daily Churn Risk Scoring
-- Run this query daily to score all active users

WITH user_features AS (
    SELECT 
        u.user_id,
        u.segment,
        DATEDIFF(CURRENT_DATE, u.account_created_date) as account_age_days,
        u.portfolio_value,
        u.n_assets,
        
        -- 30-day trading metrics
        COUNT(t.trade_id) as trades_30d,
        AVG(t.trade_amount) as avg_trade_size,
        
        -- Days since last activity
        DATEDIFF(CURRENT_DATE, MAX(u.last_login_date)) as days_since_login,
        
        -- Trading frequency trend (compare last 14 days to previous 14 days)
        (COUNT(CASE WHEN t.trade_date >= DATE_SUB(CURRENT_DATE, INTERVAL 14 DAY) THEN 1 END) -
         COUNT(CASE WHEN t.trade_date BETWEEN DATE_SUB(CURRENT_DATE, INTERVAL 28 DAY) 
                                          AND DATE_SUB(CURRENT_DATE, INTERVAL 14 DAY) THEN 1 END)
        ) / NULLIF(COUNT(CASE WHEN t.trade_date BETWEEN DATE_SUB(CURRENT_DATE, INTERVAL 28 DAY) 
                                                     AND DATE_SUB(CURRENT_DATE, INTERVAL 14 DAY) THEN 1 END), 0) 
        as trade_freq_trend,
        
        -- Feature adoption
        MAX(CASE WHEN yp.user_id IS NOT NULL THEN 1 ELSE 0 END) as yield_adoption,
        MAX(CASE WHEN at.user_id IS NOT NULL THEN 1 ELSE 0 END) as advanced_trading,
        
        -- Portfolio performance
        ((u.portfolio_value - LAG(u.portfolio_value, 30) OVER (PARTITION BY u.user_id ORDER BY snapshot_date)) 
         / NULLIF(LAG(u.portfolio_value, 30) OVER (PARTITION BY u.user_id ORDER BY snapshot_date), 0)) * 100 
        as portfolio_return_30d
        
    FROM users u
    LEFT JOIN trades t 
        ON u.user_id = t.user_id 
        AND t.trade_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
    LEFT JOIN yield_products yp 
        ON u.user_id = yp.user_id
    LEFT JOIN advanced_trading_users at 
        ON u.user_id = at.user_id
    WHERE u.status = 'active'
    GROUP BY u.user_id
),

feature_engineering AS (
    SELECT 
        *,
        -- Engagement score
        (CASE WHEN days_since_login < 7 THEN 2 ELSE 0 END +
         CASE WHEN trades_30d > 5 THEN 1 ELSE 0 END +
         yield_adoption +
         advanced_trading) as engagement_score,
        
        -- Risk flags
        CASE WHEN days_since_login > 14 
              OR trade_freq_trend < -0.3 
              OR n_assets = 1 
             THEN 1 ELSE 0 END as high_risk_profile,
        
        -- Portfolio concentration
        1.0 / n_assets as portfolio_concentration
        
    FROM user_features
),

churn_predictions AS (
    SELECT 
        user_id,
        
        -- Apply trained model coefficients (example - replace with actual model)
        -- This would come from your trained logistic regression or be a UDF for complex models
        1 / (1 + EXP(-(
            -2.5 +  -- intercept
            days_since_login * 0.15 +
            trade_freq_trend * (-0.8) +
            engagement_score * (-0.5) +
            high_risk_profile * 1.2 +
            portfolio_concentration * 0.3 +
            yield_adoption * (-0.4)
        ))) as churn_probability,
        
        -- Risk tier
        CASE 
            WHEN days_since_login > 14 OR trade_freq_trend < -0.3 THEN 'high'
            WHEN days_since_login > 7 OR engagement_score < 2 THEN 'medium'
            ELSE 'low'
        END as risk_tier,
        
        -- All features for monitoring
        days_since_login,
        trade_freq_trend,
        engagement_score,
        trades_30d,
        yield_adoption,
        n_assets,
        portfolio_value
        
    FROM feature_engineering
)

SELECT 
    user_id,
    churn_probability,
    risk_tier,
    days_since_login,
    engagement_score,
    CURRENT_TIMESTAMP as scored_at
FROM churn_predictions
ORDER BY churn_probability DESC;
"""

HIGH_RISK_USER_TARGETING = """
-- Identify High-Risk Users for Intervention Campaigns
-- Target users with high churn probability for re-engagement

SELECT 
    cs.user_id,
    u.email,
    u.segment,
    cs.churn_probability,
    cs.risk_tier,
    cs.days_since_login,
    cs.trades_30d,
    cs.portfolio_value,
    
    -- Recommended intervention
    CASE 
        WHEN cs.days_since_login BETWEEN 7 AND 14 
             AND cs.yield_adoption = 0 
             AND cs.portfolio_value > 1000
            THEN 'yield_product_education'
        
        WHEN cs.days_since_login BETWEEN 7 AND 14
            THEN 'general_reengagement'
        
        WHEN cs.n_assets = 1 AND cs.portfolio_value > 500
            THEN 'diversification_education'
        
        WHEN cs.trade_freq_trend < -0.5
            THEN 'market_insights_content'
            
        ELSE 'standard_retention'
    END as recommended_intervention,
    
    -- Priority score (for resource allocation)
    (cs.churn_probability * 0.6 + 
     (cs.portfolio_value / 10000) * 0.4) as intervention_priority
    
FROM churn_scores cs
JOIN users u ON cs.user_id = u.user_id
WHERE cs.risk_tier IN ('high', 'medium')
  AND cs.scored_at >= DATE_SUB(CURRENT_DATE, INTERVAL 1 DAY)
ORDER BY intervention_priority DESC
LIMIT 10000;
"""

COHORT_PERFORMANCE_MONITORING = """
-- Monitor Churn by Cohort and Intervention
-- Track effectiveness of retention campaigns

WITH monthly_cohorts AS (
    SELECT 
        DATE_TRUNC('month', account_created_date) as cohort_month,
        user_id,
        segment
    FROM users
),

churn_by_cohort AS (
    SELECT 
        c.cohort_month,
        c.segment,
        COUNT(DISTINCT c.user_id) as cohort_size,
        COUNT(DISTINCT CASE WHEN u.status = 'churned' THEN c.user_id END) as churned_users,
        COUNT(DISTINCT CASE WHEN u.status = 'churned' THEN c.user_id END) * 1.0 / 
            COUNT(DISTINCT c.user_id) as churn_rate,
        
        -- Average lifetime (days)
        AVG(DATEDIFF(COALESCE(u.churned_date, CURRENT_DATE), u.account_created_date)) as avg_lifetime_days,
        
        -- Intervention metrics
        COUNT(DISTINCT i.user_id) as users_with_intervention,
        COUNT(DISTINCT CASE WHEN u.status = 'active' AND i.user_id IS NOT NULL THEN c.user_id END) as intervention_retained
        
    FROM monthly_cohorts c
    JOIN users u ON c.user_id = u.user_id
    LEFT JOIN intervention_log i 
        ON c.user_id = i.user_id 
        AND i.intervention_date >= u.account_created_date
    WHERE c.cohort_month >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)
    GROUP BY c.cohort_month, c.segment
)

SELECT 
    cohort_month,
    segment,
    cohort_size,
    churned_users,
    churn_rate,
    avg_lifetime_days,
    users_with_intervention,
    
    -- Intervention effectiveness
    CASE WHEN users_with_intervention > 0 
         THEN intervention_retained * 1.0 / users_with_intervention 
         ELSE NULL 
    END as intervention_retention_rate
    
FROM churn_by_cohort
ORDER BY cohort_month DESC, segment;
"""

MODEL_PERFORMANCE_MONITORING = """
-- Monitor Model Performance Over Time
-- Track prediction accuracy and calibration

WITH daily_predictions AS (
    SELECT 
        DATE(scored_at) as prediction_date,
        user_id,
        churn_probability,
        risk_tier
    FROM churn_scores
    WHERE scored_at >= DATE_SUB(CURRENT_DATE, INTERVAL 60 DAY)
),

actual_churn AS (
    SELECT 
        user_id,
        churned_date,
        1 as actual_churned
    FROM users
    WHERE churned_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
),

prediction_outcomes AS (
    SELECT 
        dp.prediction_date,
        dp.risk_tier,
        COUNT(dp.user_id) as total_predictions,
        
        -- 30-day actual churn
        COUNT(CASE WHEN ac.user_id IS NOT NULL 
                   AND DATEDIFF(ac.churned_date, dp.prediction_date) <= 30 
              THEN 1 END) as actual_churned,
        
        AVG(dp.churn_probability) as avg_predicted_prob,
        AVG(COALESCE(ac.actual_churned, 0)) as actual_churn_rate,
        
        -- Calibration (predicted vs actual)
        AVG(dp.churn_probability) - AVG(COALESCE(ac.actual_churned, 0)) as calibration_error
        
    FROM daily_predictions dp
    LEFT JOIN actual_churn ac 
        ON dp.user_id = ac.user_id
    WHERE dp.prediction_date <= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)  -- Must wait 30 days to evaluate
    GROUP BY dp.prediction_date, dp.risk_tier
)

SELECT 
    prediction_date,
    risk_tier,
    total_predictions,
    actual_churned,
    actual_churn_rate,
    avg_predicted_prob,
    calibration_error,
    
    -- Precision for high-risk tier
    CASE WHEN risk_tier = 'high' 
         THEN actual_churned * 1.0 / total_predictions 
         ELSE NULL 
    END as high_risk_precision
    
FROM prediction_outcomes
ORDER BY prediction_date DESC, risk_tier;
"""

AB_TEST_ANALYSIS = """
-- A/B Test Analysis for Retention Intervention
-- Analyze treatment vs control for churn reduction

WITH test_population AS (
    SELECT 
        user_id,
        variant,  -- 'control' or 'treatment'
        enrolled_date,
        CASE WHEN u.churned_date IS NOT NULL 
                  AND DATEDIFF(u.churned_date, enrolled_date) <= 30
             THEN 1 ELSE 0 
        END as churned_30d
    FROM ab_test_enrollment abt
    JOIN users u ON abt.user_id = u.user_id
    WHERE test_name = 'yield_education_intervention'
      AND enrolled_date >= DATE_SUB(CURRENT_DATE, INTERVAL 60 DAY)
)

SELECT 
    variant,
    COUNT(user_id) as n_users,
    SUM(churned_30d) as churned_count,
    AVG(churned_30d) as churn_rate,
    SQRT(AVG(churned_30d) * (1 - AVG(churned_30d)) / COUNT(user_id)) as std_error,
    
    -- Confidence interval (95%)
    AVG(churned_30d) - 1.96 * SQRT(AVG(churned_30d) * (1 - AVG(churned_30d)) / COUNT(user_id)) as ci_lower,
    AVG(churned_30d) + 1.96 * SQRT(AVG(churned_30d) * (1 - AVG(churned_30d)) / COUNT(user_id)) as ci_upper
    
FROM test_population
GROUP BY variant;
"""

# Export query dictionary
SQL_QUERIES = {
    'daily_scoring': DAILY_CHURN_SCORING,
    'high_risk_targeting': HIGH_RISK_USER_TARGETING,
    'cohort_monitoring': COHORT_PERFORMANCE_MONITORING,
    'model_monitoring': MODEL_PERFORMANCE_MONITORING,
    'ab_test_analysis': AB_TEST_ANALYSIS
}

def save_queries_to_file(output_dir='sql_outputs'):
    """Save all SQL queries to individual files"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for query_name, query_sql in SQL_QUERIES.items():
        filepath = os.path.join(output_dir, f'{query_name}.sql')
        with open(filepath, 'w') as f:
            f.write(query_sql)
        print(f"✓ Saved {query_name} to {filepath}")

if __name__ == "__main__":
    print("SQL Queries Module")
    print("==================")
    print("\nAvailable queries:")
    for i, query_name in enumerate(SQL_QUERIES.keys(), 1):
        print(f"{i}. {query_name}")
    
    print("\nSaving queries to files...")
    save_queries_to_file()
