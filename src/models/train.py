import importlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.utils.logger import logger

try:
    xgboost = importlib.import_module("xgboost")
    XGBoostClassifier = xgboost.XGBClassifier
except ImportError:  # pragma: no cover - executed when xgboost is unavailable
    class XGBoostClassifier(GradientBoostingClassifier):
        """Fallback implementation used when xgboost is not installed."""

        def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                     random_state=None, eval_metric=None, **kwargs):
            super().__init__(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs,
            )
            self.eval_metric = eval_metric

def initialize_models(config: dict) -> dict:
    """Initializes standard baseline algorithms using configuration specifications."""
    logger.info("Initializing baseline models with config parameters...")
    
    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=config['models']['logistic_regression']['max_iter'],
            C=config['models']['logistic_regression']['C'],
            random_state=config['data_params']['random_state']
        ),
        "SVM": SVC(
            kernel=config['models']['svm']['kernel'],
            C=config['models']['svm']['C'],
            probability=config['models']['svm']['probability'],
            random_state=config['data_params']['random_state']
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=config['models']['random_forest']['n_estimators'],
            max_depth=config['models']['random_forest']['max_depth'],
            random_state=config['data_params']['random_state']
        ),
        "XGBoost": XGBoostClassifier(
            n_estimators=config['models']['xgboost']['n_estimators'],
            learning_rate=config['models']['xgboost']['learning_rate'],
            max_depth=config['models']['xgboost']['max_depth'],
            random_state=config['data_params']['random_state'],
            eval_metric='logloss'
        )
    }
    return models

def train_all_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Trains each initialized framework using clear target arrays and validates 
    robustness using 5-Fold Stratified Cross-Validation.
    """
    trained_models = {}
    X_arr = X_train.values
    y_arr = y_train.values
    
    # Configure Stratified K-Fold to guarantee identical class ratios across slices
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        logger.info(f"Training model: {name}...")
        
        # 1. Compute 5-Fold Stratified Cross-Validation ROC-AUC scores
        try:
            cv_scores = cross_val_score(model, X_arr, y_arr, cv=cv, scoring='roc_auc')
            mean_cv = np.mean(cv_scores)
            std_cv = np.std(cv_scores)
            logger.info(f"   -> [CV STABILITY] 5-Fold ROC-AUC: {mean_cv:.4f} (+/- {std_cv:.4f})")
        except Exception as e:
            logger.warning(f"   -> Could not compute cross-validation for {name}: {e}")
        
        # 2. Fit the final training asset
        model.fit(X_arr, y_arr)
        logger.info(f"Finished training {name}.")
        trained_models[name] = model
        
    return trained_models