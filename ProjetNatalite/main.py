"""
Main entry point for the ProjetNatalite project.

This script orchestrates the full ML pipeline:
1. Dataset loading & processing
2. Model training
3. Model inference/prediction
"""

from pathlib import Path
from typing import Optional
import sys

from loguru import logger
import typer

from ProjetNatalite.config import PROCESSED_DATA_DIR, MODELS_DIR, RAW_DATA_DIR
from ProjetNatalite.modeling import train, predict
from ProjetNatalite import dataset

# Configure loguru
logger.remove(0)
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    colorize=True,
)

app = typer.Typer(help="ProjetNatalite ML Pipeline")


@app.command()
def process_data(
    input_path: Path = typer.Option(
        RAW_DATA_DIR / "fertility_rate.csv",
        help="Path to raw input data (CSV)",
    ),
    output_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "dataset.csv",
        help="Path to save processed dataset",
    ),
):
    """Process raw data into a clean dataset."""
    logger.info(f"Processing data from: {input_path}")
    dataset.main(input_path=input_path, output_path=output_path)
    logger.success(f"Data processing complete. Output: {output_path}")


@app.command()
def train_model(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features.csv",
        help="Path to feature data (CSV)",
    ),
    labels_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "labels.csv",
        help="Path to target labels (CSV)",
    ),
    model_path: Path = typer.Option(
        MODELS_DIR / "model.pkl",
        help="Path where trained model will be saved",
    ),
):
    """Train the ML model and save to pickle format."""
    logger.info("Starting model training...")
    train.main(features_path=features_path, labels_path=labels_path, model_path=model_path)
    logger.success(f"Model training complete. Saved to: {model_path}")


@app.command()
def predict_model(
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_features.csv",
        help="Path to test features (CSV)",
    ),
    model_path: Path = typer.Option(
        MODELS_DIR / "model.pkl",
        help="Path to trained model pickle file",
    ),
    predictions_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_predictions.csv",
        help="Path where predictions will be saved",
    ),
):
    """Run inference on test data using the trained model."""
    logger.info(f"Loading model from: {model_path}")
    predict.main(
        features_path=features_path,
        model_path=model_path,
        predictions_path=predictions_path,
    )
    logger.success(f"Predictions complete. Saved to: {predictions_path}")


@app.command()
def pipeline(
    raw_data_path: Path = typer.Option(
        RAW_DATA_DIR / "fertility_rate.csv",
        help="Path to raw input data",
    ),
    processed_data_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "dataset.csv",
        help="Path to processed dataset",
    ),
    features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "features.csv",
        help="Path to feature data",
    ),
    labels_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "labels.csv",
        help="Path to target labels",
    ),
    model_path: Path = typer.Option(
        MODELS_DIR / "model.pkl",
        help="Path to save trained model",
    ),
    test_features_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_features.csv",
        help="Path to test features",
    ),
    predictions_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "test_predictions.csv",
        help="Path to save predictions",
    ),
):
    """Run the complete ML pipeline: data processing → training → prediction."""
    logger.info("=" * 70)
    logger.info("🚀 Starting ProjetNatalite ML Pipeline")
    logger.info("=" * 70)

    # Step 1: Process data
    logger.info("\n📊 STEP 1: Data Processing")
    logger.info("-" * 70)
    try:
        dataset.main(input_path=raw_data_path, output_path=processed_data_path)
        logger.success("✓ Data processing complete")
    except Exception as e:
        logger.error(f"✗ Data processing failed: {e}")
        raise

    # Step 2: Train model
    logger.info("\n🤖 STEP 2: Model Training")
    logger.info("-" * 70)
    try:
        train.main(
            features_path=features_path,
            labels_path=labels_path,
            model_path=model_path,
        )
        logger.success("✓ Model training complete")
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        raise

    # Step 3: Generate predictions
    logger.info("\n🔮 STEP 3: Model Inference")
    logger.info("-" * 70)
    try:
        predict.main(
            features_path=test_features_path,
            model_path=model_path,
            predictions_path=predictions_path,
        )
        logger.success("✓ Model inference complete")
    except Exception as e:
        logger.error(f"✗ Model inference failed: {e}")
        raise

    logger.info("\n" + "=" * 70)
    logger.success("✅ ProjetNatalite ML Pipeline completed successfully!")
    logger.info("=" * 70)


@app.callback()
def callback():
    """Typer callback for global options."""
    pass


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
