import pandas as pd
from src.utils.logger import logger

def engineer_heart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Safely preserves raw clinical features for Heart Disease prediction."""
    logger.info("Engineering features for Heart Disease data...")
    df_out = df.copy()
    
    # Optional: Add a simple, safe interaction term if columns exist
    if 'age' in df_out.columns and 'trestbps' in df_out.columns:
        df_out['age_blood_pressure_ratio'] = df_out['age'] * df_out['trestbps'] / 100.0
        
    return df_out

def engineer_diabetes_features(df: pd.DataFrame) -> pd.DataFrame:
    """Safely preserves raw clinical features for Diabetes prediction."""
    logger.info("Engineering features for Diabetes data...")
    df_out = df.copy()
    
    # Optional: Add a simple, safe interaction term if columns exist
    if 'Glucose' in df_out.columns and 'BMI' in df_out.columns:
        df_out['glucose_bmi_interaction'] = df_out['Glucose'] * df_out['BMI'] / 100.0
        
    return df_out