
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
