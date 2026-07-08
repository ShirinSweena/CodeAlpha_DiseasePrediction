import yaml
from pathlib import Path
from src.utils.logger import logger

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
            logger.info("Configuration file loaded successfully.")
            return config
    except Exception as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise e