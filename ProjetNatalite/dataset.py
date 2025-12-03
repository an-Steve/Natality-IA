from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from ProjetNatalite.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def load_and_process_data(input_path: Path) -> pd.DataFrame:
    """
    Load fertility rate data and transform from wide to long format.
    
    Args:
        input_path: Path to the raw fertility_rate.csv file
        
    Returns:
        DataFrame in long format with columns: Country, Year, FertilityRate
    """
    logger.info(f"Loading data from: {input_path}")
    
    # Load raw data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} countries with {df.shape[1]} columns")
    
    # Get country column name (first column)
    country_col = df.columns[0]
    
    # Get year columns (all columns except the first one)
    year_columns = [col for col in df.columns if col != country_col]
    
    # Convert to long format
    df_long = df.melt(
        id_vars=[country_col],
        value_vars=year_columns,
        var_name='Year',
        value_name='FertilityRate'
    )
    
    # Rename country column to standard name
    df_long = df_long.rename(columns={country_col: 'Country'})
    
    # Convert Year to integer
    df_long['Year'] = df_long['Year'].astype(int)
    
    # Remove rows with missing fertility rate values
    initial_count = len(df_long)
    df_long = df_long.dropna(subset=['FertilityRate'])
    removed_count = initial_count - len(df_long)
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} rows with missing fertility rate values")
    
    # Sort by Country and Year
    df_long = df_long.sort_values(['Country', 'Year']).reset_index(drop=True)
    
    logger.success(f"Processed data: {len(df_long)} rows, {df_long['Country'].nunique()} countries, "
                   f"{df_long['Year'].nunique()} years")
    
    return df_long


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "fertility_rate.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """
    Process raw fertility rate data into a clean dataset.
    
    Transforms wide-format data (countries × years) into long format with columns:
    - Country: Name of the country
    - Year: Year of observation
    - FertilityRate: Fertility rate value
    """
    logger.info("=" * 70)
    logger.info("Starting data processing pipeline")
    logger.info("=" * 70)
    
    # Check if input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info(f"Please place your fertility_rate.csv file in: {RAW_DATA_DIR}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Process data
    df_processed = load_and_process_data(input_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    df_processed.to_csv(output_path, index=False)
    logger.success(f"Processed dataset saved to: {output_path}")
    
    # Display summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total records: {len(df_processed)}")
    logger.info(f"Countries: {df_processed['Country'].nunique()}")
    logger.info(f"Year range: {df_processed['Year'].min()} - {df_processed['Year'].max()}")
    logger.info(f"Fertility rate range: {df_processed['FertilityRate'].min():.2f} - {df_processed['FertilityRate'].max():.2f}")
    logger.info(f"Mean fertility rate: {df_processed['FertilityRate'].mean():.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
