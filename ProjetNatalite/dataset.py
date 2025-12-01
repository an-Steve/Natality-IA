from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from ProjetNatalite.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- METTRE LE DATASET ICI ----
    input_path: Path = RAW_DATA_DIR / "fertility_rate.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
