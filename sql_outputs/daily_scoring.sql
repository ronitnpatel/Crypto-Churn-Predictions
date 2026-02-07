
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
