from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import typer

from ProjetNatalite.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_fertility_trends(df: pd.DataFrame, output_dir: Path, top_n: int = 10):
    """Plot fertility rate trends for top N countries."""
    logger.info(f"Creating fertility trends plot for top {top_n} countries...")
    
    # Calculate average fertility rate per country
    country_avg = df.groupby('Country')['FertilityRate'].mean().sort_values(ascending=False)
    top_countries = country_avg.head(top_n).index.tolist()
    
    # Filter data for top countries
    df_filtered = df[df['Country'].isin(top_countries)]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for country in top_countries:
        country_data = df_filtered[df_filtered['Country'] == country]
        ax.plot(country_data['Year'], country_data['FertilityRate'], 
                marker='o', label=country, linewidth=2, markersize=4)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Fertility Rate', fontsize=12)
    ax.set_title(f'Fertility Rate Trends - Top {top_n} Countries', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'fertility_trends.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"Saved fertility trends plot to: {output_path}")


def plot_global_average(df: pd.DataFrame, output_dir: Path):
    """Plot global average fertility rate over time."""
    logger.info("Creating global average fertility rate plot...")
    
    # Calculate yearly global average
    yearly_avg = df.groupby('Year')['FertilityRate'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(yearly_avg['Year'], yearly_avg['mean'], 
            color='darkblue', linewidth=2.5, label='Global Average')
    ax.fill_between(yearly_avg['Year'], 
                     yearly_avg['mean'] - yearly_avg['std'], 
                     yearly_avg['mean'] + yearly_avg['std'],
                     alpha=0.3, color='lightblue', label='±1 Std Dev')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Fertility Rate', fontsize=12)
    ax.set_title('Global Average Fertility Rate Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'global_average.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"Saved global average plot to: {output_path}")


def plot_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot distribution of fertility rates."""
    logger.info("Creating fertility rate distribution plot...")
    
    # Create plot with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    axes[0].hist(df['FertilityRate'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Fertility Rate', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Fertility Rates', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by decade
    df['Decade'] = (df['Year'] // 10) * 10
    df.boxplot(column='FertilityRate', by='Decade', ax=axes[1])
    axes[1].set_xlabel('Decade', fontsize=12)
    axes[1].set_ylabel('Fertility Rate', fontsize=12)
    axes[1].set_title('Fertility Rate Distribution by Decade', fontsize=12, fontweight='bold')
    plt.sca(axes[1])
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / 'distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"Saved distribution plot to: {output_path}")


def plot_heatmap(df: pd.DataFrame, output_dir: Path, sample_countries: int = 30):
    """Plot heatmap of fertility rates across countries and years."""
    logger.info(f"Creating heatmap for {sample_countries} countries...")
    
    # Sample countries for visualization (too many countries make heatmap unreadable)
    country_avg = df.groupby('Country')['FertilityRate'].mean().sort_values(ascending=False)
    selected_countries = country_avg.head(sample_countries).index.tolist()
    
    # Pivot data for heatmap
    df_filtered = df[df['Country'].isin(selected_countries)]
    pivot_data = df_filtered.pivot_table(values='FertilityRate', 
                                          index='Country', 
                                          columns='Year')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Fertility Rate'})
    ax.set_title(f'Fertility Rate Heatmap - Top {sample_countries} Countries', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    
    plt.tight_layout()
    output_path = output_dir / 'heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"Saved heatmap to: {output_path}")


def plot_correlation_matrix(features_df: pd.DataFrame, output_dir: Path):
    """Plot correlation matrix of features."""
    logger.info("Creating feature correlation matrix...")
    
    # Calculate correlation matrix
    corr_matrix = features_df.corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'correlation_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.success(f"Saved correlation matrix to: {output_path}")


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    output_dir: Path = FIGURES_DIR,
):
    """
    Generate visualizations from processed data.
    
    Creates various plots including:
    - Fertility rate trends for top countries
    - Global average fertility rate over time
    - Distribution of fertility rates
    - Heatmap of fertility rates
    - Feature correlation matrix
    """
    logger.info("=" * 70)
    logger.info("Starting plot generation")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run the data processing step first: python -m ProjetNatalite.dataset")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load processed data
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records")
    
    # Generate plots
    try:
        plot_fertility_trends(df, output_dir, top_n=10)
        plot_global_average(df, output_dir)
        plot_distribution(df, output_dir)
        plot_heatmap(df, output_dir, sample_countries=30)
        
        # Generate correlation matrix if features exist
        if features_path.exists():
            logger.info(f"Loading features from: {features_path}")
            features_df = pd.read_csv(features_path)
            plot_correlation_matrix(features_df, output_dir)
        else:
            logger.warning(f"Features file not found: {features_path}. Skipping correlation matrix.")
    
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        raise
    
    logger.info("\n" + "=" * 70)
    logger.success(f"All plots generated successfully in: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
