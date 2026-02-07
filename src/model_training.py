"""
Model Training Module
Trains and compares multiple ML models for churn prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, classification_report, roc_curve,
    precision_recall_curve
)
import joblib
import os

class ChurnModelTrainer:
    """
    Trains multiple models and provides comparison
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train logistic regression baseline"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state,
            C=0.1  # Regularization strength
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = metrics
        
        self._print_metrics('Logistic Regression', metrics)
        
        return model, metrics
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train random forest ensemble"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=100,
            min_samples_leaf=50,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        self.models['random_forest'] = model
        self.results['random_forest'] = metrics
        
        self._print_metrics('Random Forest', metrics)
        
        return model, metrics
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost gradient boosting model"""
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            early_stopping_rounds=10
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        self.models['xgboost'] = model
        self.results['xgboost'] = metrics
        
        self._print_metrics('XGBoost', metrics)
        
        return model, metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auc_pr': average_precision_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
        
        # Precision at top 10% (important for targeting)
        top_10_pct_threshold = np.percentile(y_pred_proba, 90)
        top_10_pct_pred = (y_pred_proba >= top_10_pct_threshold).astype(int)
        metrics['precision_at_10pct'] = precision_score(y_true, top_10_pct_pred)
        
        # Store predictions for later analysis
        metrics['y_true'] = y_true
        metrics['y_pred'] = y_pred
        metrics['y_pred_proba'] = y_pred_proba
        
        return metrics
    
    def _print_metrics(self, model_name, metrics):
        """Print formatted metrics"""
        print(f"\n{model_name} Validation Results:")
        print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:         {metrics['auc_pr']:.4f}")
        print(f"  Precision:      {metrics['precision']:.4f}")
        print(f"  Recall:         {metrics['recall']:.4f}")
        print(f"  F1 Score:       {metrics['f1']:.4f}")
        print(f"  Precision@10%:  {metrics['precision_at_10pct']:.4f}")
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all models and compare"""
        
        # Train models
        self.train_logistic_regression(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Print comparison
        self.print_model_comparison()
        
        # Select best model
        best_model_name = self.select_best_model()
        
        return self.models, self.results, best_model_name
    
    def print_model_comparison(self):
        """Print comparison table of all models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON (Validation Set)")
        print("="*70)
        
        comparison_df = pd.DataFrame({
            model: {
                'AUC-ROC': results['auc_roc'],
                'AUC-PR': results['auc_pr'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1': results['f1'],
                'Precision@10%': results['precision_at_10pct']
            }
            for model, results in self.results.items()
        }).T
        
        print(comparison_df.round(4).to_string())
        print("="*70)
    
    def select_best_model(self, primary_metric='auc_roc'):
        """Select best model based on primary metric"""
        
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.results.items():
            score = results[primary_metric]
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        print(f"\n🏆 Best Model: {best_model_name.upper()} (AUC-ROC: {best_score:.4f})")
        
        return best_model_name
    
    def evaluate_on_test_set(self, X_test, y_test, model_name=None):
        """Final evaluation on held-out test set"""
        
        if model_name is None:
            model_name = self.select_best_model()
        
        model = self.models[model_name]
        
        print(f"\n{'='*70}")
        print(f"FINAL TEST SET EVALUATION - {model_name.upper()}")
        print(f"{'='*70}")
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self._print_metrics(f'{model_name} (Test Set)', metrics)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
        
        return metrics
    
    def save_best_model(self, output_dir='models', model_name=None):
        """Save the best model to disk"""
        
        if model_name is None:
            model_name = self.select_best_model()
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'{model_name}_best.pkl')
        
        joblib.dump(self.models[model_name], filepath)
        print(f"\n✓ Model saved to {filepath}")
        
        return filepath

def get_feature_importance(model, feature_names, model_type='xgboost', top_n=15):
    """
    Extract and rank feature importance
    """
    
    if model_type in ['xgboost', 'random_forest']:
        importances = model.feature_importances_
    elif model_type == 'logistic_regression':
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df.head(top_n)

if __name__ == "__main__":
    # This would be run from main.py, but can be tested standalone
    print("Model training module loaded successfully")
