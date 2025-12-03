from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from ProjetNatalite.config import MODELS_DIR, PROCESSED_DATA_DIR
from ProjetNatalite.classification import LinearClassifier

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a linear regression model on fertility rate features.
    
    Args:
        features_path: Path to features CSV file
        labels_path: Path to labels CSV file
        model_path: Path where trained model will be saved
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    """
    logger.info("=" * 70)
    logger.info("Starting model training")
    logger.info("=" * 70)
    
    # Check if input files exist
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.info("Please run the feature generation step first: python -m ProjetNatalite.features")
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        logger.info("Please run the feature generation step first: python -m ProjetNatalite.features")
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Load features and labels
    logger.info(f"Loading features from: {features_path}")
    X = pd.read_csv(features_path)
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    logger.info(f"Loading labels from: {labels_path}")
    y = pd.read_csv(labels_path).iloc[:, 0]
    logger.info(f"Loaded {len(y)} labels")
    
    # Initialize and train classifier
    logger.info(f"\nTraining model with:")
    logger.info(f"  - Test size: {test_size * 100:.0f}%")
    logger.info(f"  - Random state: {random_state}")
    logger.info(f"  - Normalization: True")
    
    classifier = LinearClassifier()
    metrics = classifier.train(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
        normalize=True,
    )
    
    # Create model directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    classifier.save_model(model_path)
    
    # Display training results
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 70)
    logger.info(f"Training samples: {int(metrics['n_samples'] * (1 - test_size))}")
    logger.info(f"Test samples: {int(metrics['n_samples'] * test_size)}")
    logger.info(f"Number of features: {metrics['n_features']}")
    logger.info("\nRegression Metrics:")
    logger.info(f"  Train R² Score: {metrics['train_r2']:.4f}")
    logger.info(f"  Test R² Score:  {metrics['test_r2']:.4f}")
    logger.info(f"  Train RMSE:     {metrics['train_rmse']:.4f}")
    logger.info(f"  Test RMSE:      {metrics['test_rmse']:.4f}")
    logger.info(f"  Train MAE:      {metrics['train_mae']:.4f}")
    logger.info(f"  Test MAE:       {metrics['test_mae']:.4f}")
    logger.info("=" * 70)
    
    # Display feature importance
    feature_importance = classifier.get_feature_importance()
    if feature_importance is not None:
        logger.info("\nTop 10 Most Important Features:")
        logger.info("-" * 70)
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']:35s} | Coef: {row['coefficient']:8.4f}")
        logger.info("=" * 70)
    
    logger.success(f"\nModel training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    app()
