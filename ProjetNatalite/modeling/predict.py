from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pickle
from pathlib import Path as _Path

from ProjetNatalite.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    # Try to load the model .pkl file and report
    if model_path.exists():
        with open(model_path, "rb") as f:
            model_obj = pickle.load(f)
        logger.info(f"Loaded model from {model_path}: {model_obj}")
    else:
        logger.warning(f"Model file not found at {model_path}.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
