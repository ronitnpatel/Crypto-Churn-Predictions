"""
Data Generation Module
Generates synthetic crypto trading platform user data with realistic patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_crypto_market_data(n_days=90):
    """Generate realistic crypto market volatility data"""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Simulate BTC price with realistic volatility
    btc_returns = np.random.normal(0.001, 0.04, n_days)
    btc_prices = 40000 * np.exp(np.cumsum(btc_returns))
    
    # Calculate rolling volatility
    volatility = pd.Series(btc_returns).rolling(7).std() * np.sqrt(365)
    
    market_data = pd.DataFrame({
        'date': dates,
        'btc_price': btc_prices,
        'volatility': volatility.fillna(volatility.mean())
    })
    
    return market_data

def generate_user_data(n_users=50000):
    """Generate synthetic user base with different segments"""
    
    # User segments
    segment_probs = [0.6, 0.3, 0.08, 0.02]  # casual, active, power, whale
    segments = np.random.choice(
        ['casual', 'active', 'power_user', 'whale'],
        size=n_users,
        p=segment_probs
    )
    
    # Account age (days)
    account_age = np.random.gamma(shape=2, scale=60, size=n_users)
    account_age = np.clip(account_age, 1, 730)  # Max 2 years
    
    # Generate features based on segment
    data = []
    
    for i in range(n_users):
        segment = segments[i]
        age = account_age[i]
        
        # Segment-specific parameters
        if segment == 'casual':
            base_trade_freq = np.random.uniform(0.5, 3)  # trades per week
            portfolio_size = np.random.uniform(100, 2000)
            n_assets = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            yield_adoption = np.random.random() < 0.1
            advanced_trading = np.random.random() < 0.05
            churn_base_prob = 0.35
            
        elif segment == 'active':
            base_trade_freq = np.random.uniform(3, 10)
            portfolio_size = np.random.uniform(2000, 20000)
            n_assets = np.random.choice([2, 3, 4, 5], p=[0.3, 0.4, 0.2, 0.1])
            yield_adoption = np.random.random() < 0.4
            advanced_trading = np.random.random() < 0.3
            churn_base_prob = 0.15
            
        elif segment == 'power_user':
            base_trade_freq = np.random.uniform(10, 30)
            portfolio_size = np.random.uniform(20000, 100000)
            n_assets = np.random.choice([4, 5, 6, 7, 8], p=[0.2, 0.3, 0.3, 0.1, 0.1])
            yield_adoption = np.random.random() < 0.7
            advanced_trading = np.random.random() < 0.8
            churn_base_prob = 0.05
            
        else:  # whale
            base_trade_freq = np.random.uniform(5, 20)
            portfolio_size = np.random.uniform(100000, 1000000)
            n_assets = np.random.choice([5, 6, 7, 8, 10], p=[0.2, 0.2, 0.3, 0.2, 0.1])
            yield_adoption = np.random.random() < 0.9
            advanced_trading = np.random.random() < 0.6
            churn_base_prob = 0.03
        
        # Recent activity (last 30 days)
        recent_activity_multiplier = np.random.lognormal(0, 0.5)
        trades_30d = max(0, int(base_trade_freq * 4.3 * recent_activity_multiplier))
        
        # Days since last login (key churn predictor)
        if np.random.random() < churn_base_prob:
            # Churned or churning users
            days_since_login = np.random.gamma(shape=2, scale=10)
            days_since_login = min(days_since_login, 90)
        else:
            # Active users
            days_since_login = np.random.exponential(scale=2)
            days_since_login = min(days_since_login, 14)
        
        # Trading frequency trend (declining = higher churn)
        trade_freq_trend = np.random.normal(0, 0.3)
        if days_since_login > 7:
            trade_freq_trend = np.random.normal(-0.5, 0.2)  # Declining
        
        # Market volatility response (panic sellers churn more)
        volatility_sensitivity = np.random.normal(0, 1)
        
        # Average trade size
        avg_trade_size = portfolio_size * np.random.uniform(0.05, 0.3)
        
        # Portfolio performance (last 30 days, %)
        portfolio_return_30d = np.random.normal(2, 15)
        
        # Determine actual churn (target variable)
        # Churn probability influenced by multiple factors
        churn_prob = churn_base_prob
        
        # Days since login is strongest predictor
        if days_since_login > 14:
            churn_prob += 0.4
        elif days_since_login > 7:
            churn_prob += 0.2
        
        # Declining activity
        if trade_freq_trend < -0.3:
            churn_prob += 0.15
        
        # Low engagement features
        if not yield_adoption:
            churn_prob += 0.1
        if n_assets == 1:
            churn_prob += 0.1
        
        # Poor performance
        if portfolio_return_30d < -10:
            churn_prob += 0.1
        
        churn_prob = min(churn_prob, 0.95)
        churned = np.random.random() < churn_prob
        
        data.append({
            'user_id': f'user_{i:06d}',
            'segment': segment,
            'account_age_days': int(age),
            'portfolio_value': round(portfolio_size, 2),
            'n_assets': n_assets,
            'trades_30d': trades_30d,
            'avg_trade_size': round(avg_trade_size, 2),
            'days_since_login': round(days_since_login, 1),
            'trade_freq_trend': round(trade_freq_trend, 3),
            'yield_adoption': int(yield_adoption),
            'advanced_trading': int(advanced_trading),
            'volatility_sensitivity': round(volatility_sensitivity, 3),
            'portfolio_return_30d': round(portfolio_return_30d, 2),
            'churned': int(churned)
        })
    
    df = pd.DataFrame(data)
    
    return df

def add_derived_features(df):
    """Add additional engineered features"""
    
    # Trading intensity (trades per week normalized by account age)
    df['trading_intensity'] = (df['trades_30d'] / 4.3) / np.log1p(df['account_age_days'] / 7)
    
    # Portfolio concentration (inverse of diversity)
    df['portfolio_concentration'] = 1 / df['n_assets']
    
    # Engagement score (composite)
    df['engagement_score'] = (
        (df['days_since_login'] < 7).astype(int) * 2 +
        (df['trades_30d'] > 5).astype(int) +
        df['yield_adoption'] +
        df['advanced_trading']
    )
    
    # Risk profile
    df['high_risk_profile'] = (
        (df['days_since_login'] > 14) |
        (df['trade_freq_trend'] < -0.3) |
        (df['n_assets'] == 1)
    ).astype(int)
    
    # Account maturity stage
    df['account_stage'] = pd.cut(
        df['account_age_days'],
        bins=[0, 30, 90, 180, 365, 730],
        labels=['new', 'growing', 'established', 'mature', 'veteran']
    )
    
    return df

def save_data(df, filepath='data/crypto_users.csv'):
    """Save generated data"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print(f"\nSegment distribution:")
    print(df['segment'].value_counts(normalize=True))

if __name__ == "__main__":
    print("Generating crypto trading platform user data...")
    
    # Generate market data
    market_data = generate_crypto_market_data()
    print(f"Generated {len(market_data)} days of market data")
    
    # Generate user data
    users_df = generate_user_data(n_users=50000)
    
    # Add derived features
    users_df = add_derived_features(users_df)
    
    # Save
    save_data(users_df)
    
    print("\nSample of generated data:")
    print(users_df.head(10))
