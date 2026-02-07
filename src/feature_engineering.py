"""
Feature Engineering Module
Prepares features for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def create_feature_matrix(df):
    """
    Create feature matrix from raw data
    
    Returns:
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
    """
    
    # Separate features and target
    target_col = 'churned'
    id_cols = ['user_id']
    categorical_cols = ['segment', 'account_stage']
    
    # Create copy to avoid modifying original
    df_features = df.copy()
    
    # Encode categorical variables
    le_segment = LabelEncoder()
    df_features['segment_encoded'] = le_segment.fit_transform(df_features['segment'])
    
    le_stage = LabelEncoder()
    df_features['account_stage_encoded'] = le_stage.fit_transform(df_features['account_stage'])
    
    # Select features for modeling
    feature_cols = [
        # Core trading behavior
        'trades_30d',
        'avg_trade_size',
        'trading_intensity',
        'trade_freq_trend',
        
        # Account characteristics
        'account_age_days',
        'portfolio_value',
        'n_assets',
        'portfolio_concentration',
        
        # Engagement metrics
        'days_since_login',
        'engagement_score',
        'yield_adoption',
        'advanced_trading',
        
        # Market interaction
        'volatility_sensitivity',
        'portfolio_return_30d',
        
        # Derived features
        'high_risk_profile',
        
        # Encoded categoricals
        'segment_encoded',
        'account_stage_encoded'
    ]
    
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    return X, y, feature_cols

def create_train_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train, validation, and test sets
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    """
    Standardize features using training set statistics
    
    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def engineer_interaction_features(df):
    """
    Create interaction features that might be predictive
    """
    
    df_eng = df.copy()
    
    # Interaction: high value but low engagement
    df_eng['value_engagement_ratio'] = df_eng['portfolio_value'] / (df_eng['engagement_score'] + 1)
    
    # Interaction: recent inactivity with declining trend
    df_eng['inactivity_decline_risk'] = (
        df_eng['days_since_login'] * np.abs(df_eng['trade_freq_trend'].clip(upper=0))
    )
    
    # Interaction: portfolio concentration × poor performance
    df_eng['concentration_loss'] = (
        df_eng['portfolio_concentration'] * df_eng['portfolio_return_30d'].clip(upper=0).abs()
    )
    
    return df_eng

def create_user_segments_for_analysis(df):
    """
    Create user segments for cohort analysis
    """
    
    df_seg = df.copy()
    
    # Risk-based segmentation
    conditions = [
        (df_seg['days_since_login'] > 14) | (df_seg['trade_freq_trend'] < -0.3),
        (df_seg['days_since_login'] > 7) | (df_seg['engagement_score'] < 2),
        True
    ]
    choices = ['high_risk', 'medium_risk', 'low_risk']
    df_seg['risk_segment'] = np.select(conditions, choices, default='low_risk')
    
    # Value-based segmentation
    df_seg['value_segment'] = pd.cut(
        df_seg['portfolio_value'],
        bins=[0, 1000, 10000, 50000, np.inf],
        labels=['small', 'medium', 'large', 'whale']
    )
    
    return df_seg

def prepare_data_for_modeling(filepath='data/crypto_users.csv'):
    """
    Complete data preparation pipeline
    
    Returns:
        Dictionary containing all prepared datasets and metadata
    """
    
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print("Engineering additional features...")
    df = engineer_interaction_features(df)
    
    print("Creating feature matrix...")
    X, y, feature_names = create_feature_matrix(df)
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)
    
    print("Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    
    print("Creating user segments...")
    df_segments = create_user_segments_for_analysis(df)
    
    # Package everything
    data_package = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'df_full': df,
        'df_segments': df_segments
    }
    
    print(f"\nData preparation complete!")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_names)}")
    print(f"Churn rate - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}, Test: {y_test.mean():.2%}")
    
    return data_package

if __name__ == "__main__":
    # Test the feature engineering pipeline
    data = prepare_data_for_modeling()
    
    print("\nFeature names:")
    for i, feat in enumerate(data['feature_names'], 1):
        print(f"{i}. {feat}")
