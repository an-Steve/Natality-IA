import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import tempfile
import sys

# Ajouter le répertoire parent au chemin
sys.path.insert(0, str(Path(__file__).parent.parent))

from ProjetNatalite.dataset import main
from ProjetNatalite.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class TestDatasetMain:
    """Tests pour la fonction main du dataset"""
    
    def test_main_function_exists(self):
        """Test que la fonction main existe"""
        assert callable(main)
    
    @patch('ProjetNatalite.dataset.logger')
    @patch('ProjetNatalite.dataset.tqdm')
    def test_main_logs_processing_start(self, mock_tqdm, mock_logger):
        """Test que la fonction log le début du traitement"""
        mock_tqdm.return_value = range(10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.csv"
            input_path.write_text("col1,col2\n1,2\n")
            
            main(input_path=input_path, output_path=output_path)
            
            # Vérifier que logger.info a été appelé
            assert mock_logger.info.called
            # Vérifier que le message de démarrage est correct
            assert any("Processing dataset" in str(call) for call in mock_logger.info.call_args_list)
    
    @patch('ProjetNatalite.dataset.logger')
    @patch('ProjetNatalite.dataset.tqdm')
    def test_main_logs_success(self, mock_tqdm, mock_logger):
        """Test que la fonction log le succès"""
        mock_tqdm.return_value = range(10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.csv"
            input_path.write_text("col1,col2\n1,2\n")
            
            main(input_path=input_path, output_path=output_path)
            
            # Vérifier que logger.success a été appelé
            assert mock_logger.success.called
            assert any("complete" in str(call) for call in mock_logger.success.call_args_list)
    
    @patch('ProjetNatalite.dataset.logger')
    @patch('ProjetNatalite.dataset.tqdm')
    def test_main_iteration_5_special_log(self, mock_tqdm, mock_logger):
        """Test que la fonction log un message spécial à l'itération 5"""
        mock_tqdm.return_value = range(10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.csv"
            input_path.write_text("col1,col2\n1,2\n")
            
            main(input_path=input_path, output_path=output_path)
            
            # Vérifier que le message pour l'itération 5 est présent
            assert any("iteration 5" in str(call) for call in mock_logger.info.call_args_list)
    
    @patch('ProjetNatalite.dataset.logger')
    def test_main_with_real_files(self, mock_logger):
        """Test que la fonction fonctionne avec des fichiers réels"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.csv"
            
            # Créer un fichier CSV test
            input_path.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
            
            # Exécuter la fonction
            try:
                main(input_path=input_path, output_path=output_path)
                assert True  # La fonction s'est exécutée sans erreur
            except Exception as e:
                pytest.fail(f"main() a levé une exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
