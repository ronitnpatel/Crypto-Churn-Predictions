
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
