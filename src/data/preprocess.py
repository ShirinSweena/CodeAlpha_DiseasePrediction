import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.logger import logger

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detects and handles missing values in the dataset."""
    logger.info("Checking for missing values...")
    df_clean = df.copy()
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values. Dropping rows with missing data.")
        df_clean = df_clean.dropna()
    else:
        logger.info("No missing values detected.")
    
    # CRITICAL FIX: Ensure indices are completely defragmented and aligned after cleaning
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def split_features_target(df: pd.DataFrame, target_column: str):
    """Splits the DataFrame into feature matrix X and target vector y with absolute index alignment."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")
    
    # Force alignment on the base DataFrame first
    df_aligned = df.reset_index(drop=True)
    
    X = df_aligned.drop(columns=[target_column])
    y = df_aligned[target_column]
    return X, y

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Safely normalizes continuous metrics without blowing out evaluation distributions or dropping indices."""
    logger.info("Applying StandardScaler to features...")
    
    # Force reset tracking indices to prevent alignment drifts during matrix array slicing
    X_train_clean = X_train.reset_index(drop=True)
    X_test_clean = X_test.reset_index(drop=True)
    
    cols_to_scale = [
        col for col in X_train_clean.columns 
        if X_train_clean[col].nunique() > 5
    ]
    
    if cols_to_scale:
        scaler = StandardScaler()
        X_train_clean[cols_to_scale] = scaler.fit_transform(X_train_clean[cols_to_scale])
        X_test_clean[cols_to_scale] = scaler.transform(X_test_clean[cols_to_scale])
    else:
        scaler = None
        
    return X_train_clean, X_test_clean, scaler