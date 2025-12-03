from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from ProjetNatalite.config import MODELS_DIR, PROCESSED_DATA_DIR
from ProjetNatalite.classification import LinearClassifier

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """
    Run inference on test data using trained model.
    
    Args:
        features_path: Path to test features CSV file
        model_path: Path to trained model pickle file
        predictions_path: Path where predictions will be saved
    """
    logger.info("=" * 70)
    logger.info("Starting model inference")
    logger.info("=" * 70)
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please train the model first: python -m ProjetNatalite.modeling.train")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check if test features exist
    if not features_path.exists():
        logger.warning(f"Test features file not found: {features_path}")
        logger.info("Using training features for demonstration purposes")
        features_path = PROCESSED_DATA_DIR / "features.csv"
        
        if not features_path.exists():
            logger.error(f"No feature files found at: {PROCESSED_DATA_DIR}")
            raise FileNotFoundError(f"No feature files found")
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    classifier = LinearClassifier(model_path)
    logger.success("Model loaded successfully")
    
    # Load test features
    logger.info(f"Loading test features from: {features_path}")
    X_test = pd.read_csv(features_path)
    logger.info(f"Loaded {X_test.shape[0]} samples with {X_test.shape[1]} features")
    
    # Make predictions
    logger.info("Generating predictions...")
    predictions = classifier.predict(X_test)
    logger.success(f"Generated {len(predictions)} predictions")
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'predicted_fertility_rate': predictions
    })
    
    # Create output directory if it doesn't exist
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_df.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to: {predictions_path}")
    
    # Display prediction summary
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Number of predictions: {len(predictions)}")
    logger.info(f"Mean predicted value:  {predictions.mean():.4f}")
    logger.info(f"Std predicted value:   {predictions.std():.4f}")
    logger.info(f"Min predicted value:   {predictions.min():.4f}")
    logger.info(f"Max predicted value:   {predictions.max():.4f}")
    logger.info("=" * 70)
    
    logger.info("\nFirst 10 predictions:")
    logger.info("-" * 70)
    for i, pred in enumerate(predictions[:10]):
        logger.info(f"  Sample {i+1:3d}: {pred:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
