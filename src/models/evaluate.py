import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from src.utils.logger import logger

def evaluate_predictions(trained_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluates trained configurations cleanly by finding the mathematically optimal threshold for each model."""
    logger.info("Evaluating models on test metrics...")
    
    results = []
    y_true = np.array(y_test).flatten()
    X_test_arr = X_test.values

    for name, model in trained_models.items():
        # 1. Pull probabilities cleanly
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_arr)[:, 1]
        elif hasattr(model, "decision_function"):
            decision_scores = model.decision_function(X_test_arr)
            probs = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-9)
        else:
            probs = model.predict(X_test_arr)

        # 2. Automatically optimize the decision threshold if probabilities are available
        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            fpr, tpr, thresholds = roc_curve(y_true, probs)
            # Find the threshold that maximizes the geometric mean of sensitivity and specificity
            gmeans = np.sqrt(tpr * (1 - fpr))
            best_idx = np.argmax(gmeans)
            optimal_threshold = thresholds[best_idx]
            
            # Bound the threshold to reasonable ranges to prevent edge-case collapses
            optimal_threshold = np.clip(optimal_threshold, 0.25, 0.50)
            
            preds = (probs >= optimal_threshold).astype(int)
            logger.info(f"Optimized threshold for {name}: {optimal_threshold:.4f}")
        else:
            preds = model.predict(X_test_arr)
            
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        roc = roc_auc_score(y_true, probs)
        
        logger.info(f"Results for {name} -> Acc: {acc:.4f} | F1: {f1:.4f}")
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc
        })
        
    return pd.DataFrame(results)