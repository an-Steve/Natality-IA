from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
import typer

from ProjetNatalite.config import PROCESSED_DATA_DIR

app = typer.Typer()


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create features and labels from processed dataset.
    
    Args:
        df: DataFrame with columns [Country, Year, FertilityRate]
        
    Returns:
        Tuple of (features_df, labels_series)
    """
    logger.info("Creating features from dataset...")
    
    # Sort by country and year
    df = df.sort_values(['Country', 'Year']).reset_index(drop=True)
    
    # Create lag features (previous years' fertility rates)
    df['FertilityRate_lag1'] = df.groupby('Country')['FertilityRate'].shift(1)
    df['FertilityRate_lag2'] = df.groupby('Country')['FertilityRate'].shift(2)
    df['FertilityRate_lag3'] = df.groupby('Country')['FertilityRate'].shift(3)
    
    # Create rolling statistics
    df['FertilityRate_rolling_mean_3'] = df.groupby('Country')['FertilityRate'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['FertilityRate_rolling_std_3'] = df.groupby('Country')['FertilityRate'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )
    
    # Create change features
    df['FertilityRate_change'] = df.groupby('Country')['FertilityRate'].diff()
    df['FertilityRate_pct_change'] = df.groupby('Country')['FertilityRate'].pct_change()
    
    # Add time-based features
    df['Year_normalized'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())
    
    # Country encoding (can be extended with more sophisticated encodings)
    df['Country_encoded'] = pd.Categorical(df['Country']).codes
    
    # Calculate country statistics
    country_stats = df.groupby('Country')['FertilityRate'].agg(['mean', 'std', 'min', 'max'])
    country_stats.columns = [f'Country_{col}_fertility' for col in country_stats.columns]
    df = df.merge(country_stats, left_on='Country', right_index=True)
    
    # Drop rows with NaN values (from lag features)
    df_clean = df.dropna()
    
    logger.info(f"Created features with {len(df_clean)} samples after removing NaN values")
    
    # Define feature columns and target
    feature_columns = [
        'Year',
        'Year_normalized',
        'Country_encoded',
        'FertilityRate_lag1',
        'FertilityRate_lag2',
        'FertilityRate_lag3',
        'FertilityRate_rolling_mean_3',
        'FertilityRate_rolling_std_3',
        'FertilityRate_change',
        'FertilityRate_pct_change',
        'Country_mean_fertility',
        'Country_std_fertility',
        'Country_min_fertility',
        'Country_max_fertility',
    ]
    
    X = df_clean[feature_columns]
    y = df_clean['FertilityRate']
    
    logger.success(f"Features created: {X.shape[1]} features, {len(X)} samples")
    
    return X, y


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    features_output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_output_path: Path = PROCESSED_DATA_DIR / "labels.csv",
):
    """
    Generate features from processed dataset for model training.
    
    Creates lagged features, rolling statistics, and other derived features
    from the fertility rate time series data.
    """
    logger.info("=" * 70)
    logger.info("Starting feature generation")
    logger.info("=" * 70)
    
    # Check if input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run the data processing step first: python -m ProjetNatalite.dataset")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load processed data
    logger.info(f"Loading processed data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Create features
    X, y = create_features(df)
    
    # Create output directory if it doesn't exist
    features_output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save features and labels
    X.to_csv(features_output_path, index=False)
    y.to_csv(labels_output_path, index=False, header=['FertilityRate'])
    
    logger.success(f"Features saved to: {features_output_path}")
    logger.success(f"Labels saved to: {labels_output_path}")
    
    # Display feature summary
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Number of features: {X.shape[1]}")
    logger.info(f"Number of samples: {len(X)}")
    logger.info(f"\nFeature names:")
    for col in X.columns:
        logger.info(f"  - {col}")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
