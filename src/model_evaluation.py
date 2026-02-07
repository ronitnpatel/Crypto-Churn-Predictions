"""
Model Evaluation and Visualization Module
Creates comprehensive visualizations and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import shap
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization
    """
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_model_comparison(self, results_dict):
        """
        Compare all models across key metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['auc_roc', 'auc_pr', 'precision_at_10pct', 'f1']
        metric_names = ['AUC-ROC', 'AUC-PR', 'Precision@10%', 'F1 Score']
        
        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            models = list(results_dict.keys())
            scores = [results_dict[m][metric] for m in models]
            
            bars = ax.bar(models, scores, color=['#3498db', '#2ecc71', '#e74c3c'])
            ax.set_ylabel(name, fontsize=12)
            ax.set_ylim([0, 1])
            ax.set_title(name, fontsize=13, fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            # Rotate x-axis labels
            ax.set_xticklabels(models, rotation=45, ha='right')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison to {filepath}")
        plt.close()
    
    def plot_roc_curves(self, results_dict):
        """
        Plot ROC curves for all models
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
            auc = results['auc_roc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})',
                    color=colors[idx], linewidth=2.5)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=1.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves to {filepath}")
        plt.close()
    
    def plot_precision_recall_curves(self, results_dict):
        """
        Plot precision-recall curves for all models
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for idx, (model_name, results) in enumerate(results_dict.items()):
            precision, recall, _ = precision_recall_curve(
                results['y_true'], results['y_pred_proba']
            )
            auc_pr = results['auc_pr']
            
            plt.plot(recall, precision, label=f'{model_name} (AUC-PR={auc_pr:.3f})',
                    color=colors[idx], linewidth=2.5)
        
        # Baseline (random classifier)
        baseline = results['y_true'].mean()
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Baseline ({baseline:.3f})', linewidth=1.5)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, 'precision_recall_curves.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved precision-recall curves to {filepath}")
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, model_name='xgboost', top_n=15):
        """
        Plot feature importance
        """
        from model_training import get_feature_importance
        
        importance_df = get_feature_importance(model, feature_names, model_name, top_n)
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))
        
        plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances - {model_name.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'feature_importance_{model_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance to {filepath}")
        plt.close()
        
        return importance_df
    
    def plot_shap_summary(self, model, X_sample, feature_names, model_name='xgboost'):
        """
        Create SHAP summary plot for model interpretability
        """
        print(f"\nGenerating SHAP values for {model_name}...")
        
        # Sample data if too large
        if len(X_sample) > 1000:
            sample_indices = np.random.choice(len(X_sample), 1000, replace=False)
            X_shap = X_sample.iloc[sample_indices]
        else:
            X_shap = X_sample
        
        # Create explainer
        if model_name == 'xgboost':
            explainer = shap.TreeExplainer(model)
        elif model_name == 'random_forest':
            explainer = shap.TreeExplainer(model)
        else:  # logistic regression
            explainer = shap.LinearExplainer(model, X_shap)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_shap)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, feature_names=feature_names, 
                         show=False, plot_size=(10, 8))
        plt.title(f'SHAP Feature Impact - {model_name.upper()}', 
                 fontsize=14, fontweight='bold', pad=20)
        
        filepath = os.path.join(self.output_dir, f'shap_summary_{model_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved SHAP summary to {filepath}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model'):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Retained', 'Churned'],
                   yticklabels=['Retained', 'Churned'])
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        filepath = os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {filepath}")
        plt.close()
    
    def plot_churn_by_segment(self, df_segments):
        """
        Analyze churn rates by different segments
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Churn Analysis by User Segments', fontsize=16, fontweight='bold')
        
        # By user segment
        ax = axes[0, 0]
        churn_by_segment = df_segments.groupby('segment')['churned'].mean().sort_values()
        churn_by_segment.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Churn Rate', fontsize=11)
        ax.set_title('Churn Rate by User Segment', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        for i, v in enumerate(churn_by_segment):
            ax.text(v + 0.01, i, f'{v:.1%}', va='center')
        
        # By risk segment
        ax = axes[0, 1]
        churn_by_risk = df_segments.groupby('risk_segment')['churned'].mean()
        churn_by_risk = churn_by_risk.reindex(['low_risk', 'medium_risk', 'high_risk'])
        churn_by_risk.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
        ax.set_xlabel('Risk Segment', fontsize=11)
        ax.set_ylabel('Churn Rate', fontsize=11)
        ax.set_title('Churn Rate by Risk Segment', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim([0, 1])
        for i, v in enumerate(churn_by_risk):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center')
        
        # By yield adoption
        ax = axes[1, 0]
        churn_by_yield = df_segments.groupby('yield_adoption')['churned'].mean()
        churn_by_yield.plot(kind='bar', ax=ax, color=['red', 'green'])
        ax.set_xlabel('Yield Product Adoption', fontsize=11)
        ax.set_ylabel('Churn Rate', fontsize=11)
        ax.set_title('Churn Rate by Yield Adoption', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['No', 'Yes'], rotation=0)
        ax.set_ylim([0, 1])
        for i, v in enumerate(churn_by_yield):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center')
        
        # By portfolio diversity
        ax = axes[1, 1]
        churn_by_assets = df_segments.groupby('n_assets')['churned'].mean()
        churn_by_assets.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_xlabel('Number of Assets', fontsize=11)
        ax.set_ylabel('Churn Rate', fontsize=11)
        ax.set_title('Churn Rate by Portfolio Diversity', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'churn_by_segment.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved segment analysis to {filepath}")
        plt.close()
    
    def create_all_visualizations(self, models, results, data_package, best_model_name):
        """
        Generate all visualization outputs
        """
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Model comparison charts
        self.plot_model_comparison(results)
        self.plot_roc_curves(results)
        self.plot_precision_recall_curves(results)
        
        # Best model deep dive
        best_model = models[best_model_name]
        
        # Feature importance
        self.plot_feature_importance(
            best_model, 
            data_package['feature_names'], 
            best_model_name
        )
        
        # SHAP analysis
        self.plot_shap_summary(
            best_model,
            data_package['X_test'],
            data_package['feature_names'],
            best_model_name
        )
        
        # Confusion matrix
        y_pred = best_model.predict(data_package['X_test'])
        self.plot_confusion_matrix(
            data_package['y_test'],
            y_pred,
            best_model_name
        )
        
        # Segment analysis
        self.plot_churn_by_segment(data_package['df_segments'])
        
        print(f"\n✓ All visualizations saved to {self.output_dir}/")

if __name__ == "__main__":
    print("Model evaluation module loaded successfully")
