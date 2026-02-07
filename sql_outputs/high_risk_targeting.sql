
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
