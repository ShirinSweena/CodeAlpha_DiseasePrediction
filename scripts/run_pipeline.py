import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure the root directory is on the system path for seamless module imports
root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.config import load_config
from src.utils.logger import logger
from src.data.load_data import load_csv_data
from src.data.preprocess import handle_missing_values, split_features_target, scale_features
from src.data.feature_engineering import engineer_heart_features, engineer_diabetes_features
from src.models.train import initialize_models, train_all_models
from src.models.evaluate import evaluate_predictions
from src.explainability.shap_analysis import generate_shap_plots

def run_disease_pipeline(disease_name: str, config: dict):
    """Executes the end-to-end processing, training, and evaluation lifecycle."""
    logger.info(f"================ STARTING PIPELINE: {disease_name.upper()} ================")
    
    # 1. Resolve exact target parameters from configuration file
    if disease_name.lower() == "heart":
        data_path = config['paths']['raw_heart_data']
        feature_engineer_fn = engineer_heart_features
        target_column = "target"
    elif disease_name.lower() == "diabetes":
        data_path = config['paths']['raw_diabetes_data']
        feature_engineer_fn = engineer_diabetes_features
        target_column = "Outcome"
    else:
        raise ValueError(f"Unsupported disease pipeline parameter: {disease_name}")

    # 2. Ingest Dataset DataFrame from Raw Directories
    df_raw = load_csv_data(data_path)
    
    # 3. Clean and Isolate Matrix Invariants
    df_clean = handle_missing_values(df_raw)
    df_engineered = feature_engineer_fn(df_clean)
    X, y = split_features_target(df_engineered, target_column=target_column)
    
    # 4. Partition Arrays with Explicit Stratification to Guarantee Balanced Split Signals
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data_params']['test_size'],
        random_state=config['data_params']['random_state'],
        stratify=y  # Essential to lock exact label distribution and break the 50% wall
    )
    
    # 5. Normalize Matrix Features Safely
    X_train, X_test, scaler = scale_features(X_train_raw, X_test_raw)
    
    # 6. Initialize and Train Base Classifiers
    base_models = initialize_models(config)
    trained_models = train_all_models(base_models, X_train, y_train)
    
    # 7. Evaluate and Print Performance Summaries
    metrics_summary_df = evaluate_predictions(trained_models, X_test, y_test)
    
    print(f"\n--- Performance Scores Summary ({disease_name.upper()}) ---")
    print(metrics_summary_df.to_string(index=False))
    print("-" * 50 + "\n")
    
    # 8. Render Feature Attributions
    if "XGBoost" in trained_models:
        try:
            generate_shap_plots(trained_models["XGBoost"], X_train, disease_name)
        except Exception as e:
            logger.warning(f"SHAP explanation generation skipped or met an exception: {e}")

    logger.info(f"================ PIPELINE RUN COMPLETE: {disease_name.upper()} ================\n")

if __name__ == "__main__":
    # Load configuration structures globally
    project_config = load_config()
    
    # Execute sequential evaluations across both target clinical cohorts
    run_disease_pipeline("heart", project_config)
    run_disease_pipeline("diabetes", project_config)