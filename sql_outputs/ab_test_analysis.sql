
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
