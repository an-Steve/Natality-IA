import pytest
from pathlib import Path
from ProjetNatalite.cli import main   


def test_main_runs(tmp_path):
    """
    Test minimal : vérifie que la fonction main() s'exécute sans erreur.
    """

    # Création d’un faux fichier CSV TEMPORAIRE DANS tmp_path
    fake_input = tmp_path / "fertility_rate.csv"
    fake_input.write_text("col1,col2\n1,2\n3,4")

    # Chemin pour enregistrer le fichier de sortie
    fake_output = tmp_path / "dataset.csv"

    # Appel de la fonction main()
    try:
        main(
            input_path=fake_input,
            output_path=fake_output
        )
    except Exception as e:
        pytest.fail(f"La fonction main a levé une erreur : {e}")

    # Vérifie que le fichier de sortie a été créé
    assert fake_output.exists(), "Le fichier de sortie n'a pas été créé"

    # Le test passe si aucun crash et que le fichier existe

