from pathlib import Path
import pandas as pd

from loguru import logger
from tqdm import tqdm
import typer

# Assurez-vous que le fichier config.py est chargé correctement
from ProjetNatalite.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
# Importation hypothétique des fonctions de nettoyage
# Assurez-vous que ce module 'data_cleaner' et ses fonctions existent réellement.
from ProjetNatalite.pretraitement.data_cleaner import remove_nan_rows, remove_nan_cols, remove_duplicates, normalize

app = typer.Typer()


def data_loder(input_path: Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Dataset loaded from {input_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Fichier non trouvé à: {input_path}. Création d'un DataFrame de simulation.")
        # DataFrame de simulation si le fichier n'est pas trouvé
        return pd.DataFrame({
            'annee': range(2000, 2010),
            'brut_natalite': [10.5, 10.2, 9.8, 11.0, 11.1, 10.9, 10.7, 10.6, 10.5, 10.3]
        })


def data_saver(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to a CSV file."""
    # S'assurer que le répertoire de sortie existe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset saved successfully to: {output_path} with shape {df.shape}")
    return None


def load_and_clean_data(input_path: Path) -> pd.DataFrame:
    """Load and clean dataset."""
    df = data_loder(input_path)
    
    logger.info("Starting data cleaning and normalization steps...")
    
    # 1. Étapes de nettoyage
    df = remove_nan_rows(df)
    df = remove_nan_cols(df)
    df = remove_duplicates(df)
    
    # --- CORRECTIF POUR L'ERREUR DE NORMALISATION ---
    
    # 2. Isoler la colonne non numérique ('Country') avant la normalisation
    if 'Country' in df.columns:
        # Stocker la colonne 'Country' pour la remettre plus tard
        country_data = df['Country']
        # Créer un DataFrame ne contenant que les colonnes numériques (les années)
        df_numeric = df.drop(columns=['Country'])
        logger.info("Separated 'Country' column before scaling.")
    else:
        # Si la colonne 'Country' n'existe pas, normaliser tout le DataFrame
        country_data = None
        df_numeric = df
        
    # 3. Normalisation (appliquée uniquement aux données numériques)
    # Note: On suppose ici que la fonction normalize importe et utilise StandardScaler
    df_normalized_numeric = normalize(df_numeric)
    
    # 4. Recombiner les données
    if country_data is not None:
        # Assurez-vous que l'index est conservé pour la fusion
        df_normalized_numeric['Country'] = country_data.reset_index(drop=True)
        # Remettre 'Country' comme première colonne pour la lisibilité
        cols = ['Country'] + [col for col in df_normalized_numeric.columns if col != 'Country']
        df_processed = df_normalized_numeric[cols]
    else:
        df_processed = df_normalized_numeric
        
    logger.info("Data cleaning and normalization complete.")
    return df_processed


@app.command()
def main(
    # Arguments pour Typer, avec les chemins par défaut
    input_path: Path = typer.Option(RAW_DATA_DIR / "fertility_rate.csv", help="Chemin vers le dataset brut."),
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "processed_natalite_data.csv", help="Chemin pour sauvegarder le dataset traité.")
):
    logger.info("Processing dataset...")
    
    # 1. Chargement et Nettoyage/Traitement
    df_processed = load_and_clean_data(input_path)
    
    # 2. Sauvegarde du DataFrame Traité
    data_saver(df_processed, output_path)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()

# python -m ProjetNatalite.pretraitement.dataset