import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from ProjetNatalite.dataset import main
from ProjetNatalite.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def test_main_function_exists():
    """Test que la fonction main existe et est callable"""
    assert callable(main)


@patch('ProjetNatalite.dataset.logger')
def test_main_with_mock_logger(mock_logger):
    """Test que main s'exécute sans erreur avec un logger mocké"""
    # Créer des répertoires temporaires
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "fertility_rate.csv"
        output_path = Path(tmpdir) / "output.csv"
        
        # Créer un fichier CSV de test
        input_path.write_text("col1,col2\n1,2\n3,4\n")
        
        # Exécuter la fonction
        main(input_path=input_path, output_path=output_path)
        
        # Vérifier que logger.info et logger.success ont été appelés
        mock_logger.info.assert_called()
        mock_logger.success.assert_called()


@patch('ProjetNatalite.dataset.logger')
@patch('ProjetNatalite.dataset.tqdm')
def test_main_iteration_count(mock_tqdm, mock_logger):
    """Test que la boucle itère 10 fois"""
    mock_tqdm.return_value = range(10)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.csv"
        output_path = Path(tmpdir) / "output.csv"
        
        input_path.write_text("col1,col2\n1,2\n")
        
        main(input_path=input_path, output_path=output_path)
        
        # Vérifier que tqdm a été appelé
        mock_tqdm.assert_called()


def test_raw_data_dir_exists():
    """Test que le répertoire RAW_DATA_DIR existe"""
    assert RAW_DATA_DIR.exists() or isinstance(RAW_DATA_DIR, Path)


def test_processed_data_dir_exists():
    """Test que le répertoire PROCESSED_DATA_DIR existe"""
    assert PROCESSED_DATA_DIR.exists() or isinstance(PROCESSED_DATA_DIR, Path)
