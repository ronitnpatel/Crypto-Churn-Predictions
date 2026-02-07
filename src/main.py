"""
Main Execution Script
Runs the complete churn prediction pipeline
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_generation import generate_user_data, add_derived_features, save_data
from feature_engineering import prepare_data_for_modeling
from model_training import ChurnModelTrainer
from model_evaluation import ModelEvaluator
from sql_queries import save_queries_to_file

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_pipeline():
    """
    Execute the complete ML pipeline
    """
    
    print_header("CRYPTO CHURN PREDICTION MODEL - FULL PIPELINE")
    
    print_header("STEP 1: GENERATING SYNTHETIC DATA")
    
    # Generate user data
    print("Creating 50,000 synthetic users with realistic crypto trading patterns...")
    users_df = generate_user_data(n_users=50000)
    
    # Add derived features
    users_df = add_derived_features(users_df)
    
    # Save data
    os.makedirs('data', exist_ok=True)
    save_data(users_df, 'data/crypto_users.csv')
    
    print("\n✓ Data generation complete!")
    print(f"  - Total users: {len(users_df):,}")
    print(f"  - Overall churn rate: {users_df['churned'].mean():.1%}")
    print(f"  - Features: {users_df.shape[1]}")

    print_header("STEP 2: FEATURE ENGINEERING & DATA PREPARATION")
    
    data_package = prepare_data_for_modeling('data/crypto_users.csv')
    
    print("\n✓ Feature engineering complete!")
    print(f"  - Training samples: {len(data_package['X_train']):,}")
    print(f"  - Validation samples: {len(data_package['X_val']):,}")
    print(f"  - Test samples: {len(data_package['X_test']):,}")
    print(f"  - Total features: {len(data_package['feature_names'])}")

    print_header("STEP 3: TRAINING MACHINE LEARNING MODELS")
    
    trainer = ChurnModelTrainer(random_state=42)
    
    models, results, best_model_name = trainer.train_all_models(
        data_package['X_train'],
        data_package['y_train'],
        data_package['X_val'],
        data_package['y_val']
    )

    print_header("STEP 4: FINAL TEST SET EVALUATION")
    
    test_metrics = trainer.evaluate_on_test_set(
        data_package['X_test'],
        data_package['y_test'],
        best_model_name
    )
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    trainer.save_best_model('models', best_model_name)
    
    print_header("STEP 5: GENERATING VISUALIZATIONS & INSIGHTS")
    
    evaluator = ModelEvaluator(output_dir='outputs')
    
    evaluator.create_all_visualizations(
        models,
        results,
        data_package,
        best_model_name
    )
    
    print_header("STEP 6: GENERATING SQL QUERIES")
    
    os.makedirs('sql_outputs', exist_ok=True)
    save_queries_to_file('sql_outputs')
    
    print_header("PIPELINE EXECUTION IS COMPLETE ")
    
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"\n✓ Best Model: {best_model_name.upper()}")
    print(f"  - AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"  - Precision@10%: {test_metrics['precision_at_10pct']:.4f}")
    print(f"  - F1 Score: {test_metrics['f1']:.4f}")
        
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
