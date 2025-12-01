from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pickle

from ProjetNatalite.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # Save a simple model object as a .pkl file
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_obj = {"name": "dummy_model", "version": 1}
    with open(model_path, "wb") as f:
        pickle.dump(model_obj, f)
    logger.info(f"Saved model to: {model_path}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
