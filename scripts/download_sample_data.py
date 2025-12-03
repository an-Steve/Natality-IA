"""
Script to create sample fertility rate data for testing.

If you have actual data, place it in data/raw/fertility_rate.csv instead.
This script creates synthetic sample data for demonstration purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


def create_sample_data(output_path: Path, n_countries: int = 50, start_year: int = 1960, end_year: int = 2023):
    """
    Create sample fertility rate data.
    
    Args:
        output_path: Where to save the CSV file
        n_countries: Number of countries to generate
        start_year: Starting year
        end_year: Ending year
    """
    logger.info(f"Creating sample fertility rate data...")
    logger.info(f"Countries: {n_countries}, Years: {start_year}-{end_year}")
    
    # Sample country names
    countries = [
        "Afghanistan", "Albania", "Algeria", "Argentina", "Australia",
        "Austria", "Bangladesh", "Belgium", "Brazil", "Bulgaria",
        "Canada", "Chile", "China", "Colombia", "Croatia",
        "Cuba", "Czech Republic", "Denmark", "Egypt", "Ethiopia",
        "Finland", "France", "Germany", "Ghana", "Greece",
        "Hungary", "India", "Indonesia", "Iran", "Iraq",
        "Ireland", "Israel", "Italy", "Japan", "Kenya",
        "Mexico", "Morocco", "Netherlands", "Nigeria", "Norway",
        "Pakistan", "Peru", "Poland", "Portugal", "Romania",
        "Russia", "Saudi Arabia", "South Africa", "South Korea", "Spain",
    ][:n_countries]
    
    # Generate years
    years = list(range(start_year, end_year + 1))
    
    # Create DataFrame
    data = {"Country": countries}
    
    for country in countries:
        # Generate realistic fertility rate trend for each country
        # Start with a random baseline between 2 and 7
        baseline = np.random.uniform(2.5, 7.0)
        
        # Create declining trend with some noise
        decline_rate = np.random.uniform(0.01, 0.05)  # Decline per year
        noise_level = np.random.uniform(0.05, 0.15)
        
        fertility_rates = []
        for i, year in enumerate(years):
            # General declining trend
            value = baseline - (decline_rate * i)
            
            # Add some random noise
            value += np.random.normal(0, noise_level)
            
            # Ensure reasonable bounds (0.8 to 7.5)
            value = max(0.8, min(7.5, value))
            
            fertility_rates.append(round(value, 2))
        
        # Add year columns
        for year, rate in zip(years, fertility_rates):
            if str(year) not in data:
                data[str(year)] = []
            data[str(year)].append(rate)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Randomly set some values as NaN to simulate missing data (5% of values)
    for col in df.columns[1:]:  # Skip Country column
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.success(f"Sample data created: {output_path}")
    logger.info(f"Shape: {df.shape[0]} countries × {df.shape[1]} columns")
    logger.info(f"Years covered: {start_year} to {end_year}")
    logger.info(f"Missing values: {df.isna().sum().sum()} ({df.isna().sum().sum() / df.size * 100:.1f}%)")
    
    return df


def download_real_data(output_path: Path):
    """
    Download real fertility rate data from World Bank or other sources.
    
    NOTE: You need to implement this based on your actual data source.
    """
    logger.info("Attempting to download real fertility rate data...")
    
    try:
        # Example: World Bank API (you need to install wbgapi: pip install wbgapi)
        # import wbgapi as wb
        # df = wb.data.DataFrame('SP.DYN.TFRT.IN', time=range(1960, 2024))
        # df = df.reset_index()
        # df.to_csv(output_path, index=False)
        
        logger.warning("Real data download not implemented yet.")
        logger.info("Please either:")
        logger.info("1. Place your fertility_rate.csv in data/raw/")
        logger.info("2. Implement download_real_data() function")
        logger.info("3. Use create_sample_data() for testing")
        
        return None
    
    except Exception as e:
        logger.error(f"Failed to download real data: {e}")
        return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download or create fertility rate data")
    parser.add_argument(
        "--sample", 
        action="store_true", 
        help="Create sample synthetic data instead of downloading real data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/fertility_rate.csv",
        help="Output path for the CSV file"
    )
    parser.add_argument(
        "--countries",
        type=int,
        default=50,
        help="Number of countries to generate (sample mode only)"
    )
    
    args = parser.parse_args()
    output_path = Path(args.output)
    
    if args.sample:
        logger.info("Creating sample synthetic data...")
        create_sample_data(output_path, n_countries=args.countries)
    else:
        logger.info("Attempting to download real data...")
        result = download_real_data(output_path)
        
        if result is None:
            logger.warning("Real data download failed. Creating sample data instead...")
            create_sample_data(output_path, n_countries=args.countries)
    
    logger.success(f"Data ready at: {output_path}")
    logger.info("\nNext steps:")
    logger.info("1. Run: python -m ProjetNatalite.main process-data")
    logger.info("2. Run: python -m ProjetNatalite.main pipeline")


if __name__ == "__main__":
    main()
