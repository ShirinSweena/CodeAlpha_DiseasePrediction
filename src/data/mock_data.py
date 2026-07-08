import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import logger

def generate_mock_datasets(output_dir: str = "data/raw"):
    """Generates realistic mock datasets safely using a localized RNG seed state."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Using local Generator state to avoid global side-effects across sklearn
    rng = np.random.default_rng(42)
    
    logger.info("Generating fallback mock Heart Disease dataset...")
    heart_size = 303
    heart_data = {
        'age': rng.integers(29, 78, size=heart_size),
        'sex': rng.integers(0, 2, size=heart_size),
        'cp': rng.integers(0, 4, size=heart_size),
        'trestbps': rng.integers(94, 200, size=heart_size),
        'chol': rng.integers(126, 564, size=heart_size),
        'fbs': rng.integers(0, 2, size=heart_size),
        'restecg': rng.integers(0, 3, size=heart_size),
        'thalach': rng.integers(71, 202, size=heart_size),
        'exang': rng.integers(0, 2, size=heart_size),
        'oldpeak': np.round(rng.uniform(0.0, 6.2, size=heart_size), 1),
        'slope': rng.integers(0, 3, size=heart_size),
        'ca': rng.integers(0, 5, size=heart_size),
        'thal': rng.integers(0, 4, size=heart_size),
        'target': rng.integers(0, 2, size=heart_size)
    }
    pd.DataFrame(heart_data).to_csv(f"{output_dir}/heart.csv", index=False)
    logger.info(f"Saved fallback mock heart.csv to {output_dir}")