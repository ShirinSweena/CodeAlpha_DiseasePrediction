import matplotlib.pyplot as plt
import shap
from pathlib import Path
from src.utils.logger import logger

def generate_shap_plots(model, X_test, dataset_name: str, output_dir: str = "outputs"):
    """Computes SHAP values using a tree-based model and saves the global summary plot."""
    logger.info(f"Computing SHAP feature attributions for {dataset_name} using XGBoost...")
    
    # Check if the output folder exists
    save_path = Path(output_dir) / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
        
        # Build Figure
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        
        # Save structural image
        plot_file = save_path / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        
        logger.info(f"Successfully saved SHAP summary plot to {plot_file.resolve()}")
    except Exception as e:
        logger.error(f"Failed to calculate SHAP values: {e}")