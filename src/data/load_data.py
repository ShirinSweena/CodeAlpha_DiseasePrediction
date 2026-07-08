from pathlib import Path
import pandas as pd
from src.utils.logger import logger

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Loads a CSV dataset from a given file path into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Data file not found at: {path.resolve()}")
        raise FileNotFoundError(f"Missing required file: {path}")
    
    logger.info(f"Loading data from {path.name}...")
    try:
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded {path.name} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV at {path}: {e}")
        raise e