import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
from sklearn.preprocessing import StandardScaler


from ProjetNatalite.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()



def remove_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaN values from the DataFrame."""
    initial_shape = df.shape
    df_cleaned = df.dropna()
    logger.info(f"Removed {initial_shape[0] - df_cleaned.shape[0]} rows with NaN values")
    return df_cleaned



def remove_nan_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with NaN values from the DataFrame."""
    initial_shape = df.shape
    df_cleaned = df.dropna(axis=1)
    logger.info(f"Removed {initial_shape[1] - df_cleaned.shape[1]} columns with NaN values")
    return df_cleaned



def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""
    initial_shape = df.shape
    df_cleaned = df.drop_duplicates()
    logger.info(f"Removed {initial_shape[0] - df_cleaned.shape[0]} duplicate rows")
    return df_cleaned



# Normaliser les valeurs d'un DataFrame
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the values of the DataFrame."""
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized



