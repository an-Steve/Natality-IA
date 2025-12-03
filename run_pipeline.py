"""
Script pour exécuter le pipeline complet de Natality-IA.

Ce script exécute toutes les étapes du pipeline ML:
1. Traitement des données
2. Génération des features
3. Entraînement du modèle
4. Prédictions
5. Visualisations
6. Affichage du résumé complet
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import pickle
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<level>{level: <8}</level> | <level>{message}</level>", colorize=True)


def run_command(command, description):
    """Execute une commande et affiche le résultat."""
    logger.info(f"▶ {description}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"❌ Erreur: {description}")
        logger.error(result.stderr)
        return False
    logger.success(f"✅ {description} - Terminé")
    return True


def display_summary():
    """Affiche un résumé complet des résultats."""
    logger.info("\n" + "="*80)
    logger.info("📊 RÉSUMÉ COMPLET DU PROJET NATALITY-IA")
    logger.info("="*80)
    
    # 1. Données
    dataset_path = Path("data/processed/dataset.csv")
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        logger.info("\n🗂️  DONNÉES:")
        logger.info(f"   • Pays analysés: {df['Country'].nunique()}")
        logger.info(f"   • Période: {df['Year'].min()} - {df['Year'].max()}")
        logger.info(f"   • Total d'observations: {len(df)}")
        logger.info(f"   • Taux de fertilité moyen: {df['FertilityRate'].mean():.2f}")
        logger.info(f"   • Min: {df['FertilityRate'].min():.2f}, Max: {df['FertilityRate'].max():.2f}")
    
    # 2. Features
    features_path = Path("data/processed/features.csv")
    if features_path.exists():
        features = pd.read_csv(features_path)
        logger.info("\n🔧 FEATURES:")
        logger.info(f"   • Nombre de features: {features.shape[1]}")
        logger.info(f"   • Échantillons d'entraînement: {len(features)}")
        logger.info(f"   • Features principales:")
        for col in features.columns[:5]:
            logger.info(f"      - {col}")
    
    # 3. Modèle
    model_path = Path("models/model.pkl")
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info("\n🤖 MODÈLE:")
        logger.info(f"   • Type: Linear Regression")
        logger.info(f"   • Fichier: models/model.pkl")
        logger.info(f"   • Taille: {model_path.stat().st_size / 1024:.2f} KB")
        
        # Afficher les métriques si disponibles
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            logger.info("\n📈 PERFORMANCE DU MODÈLE:")
            logger.info(f"   • R² Score (Train): {metrics.get('train_r2', 0):.4f}")
            logger.info(f"   • R² Score (Test):  {metrics.get('test_r2', 0):.4f}")
            logger.info(f"   • RMSE (Test):      {metrics.get('test_rmse', 0):.4f}")
            logger.info(f"   • MAE (Test):       {metrics.get('test_mae', 0):.4f}")
            
            # Calcul de la précision en pourcentage
            r2_test = metrics.get('test_r2', 0)
            precision_pct = r2_test * 100
            logger.info(f"\n   🎯 PRÉCISION: {precision_pct:.2f}%")
            
            if r2_test >= 0.95:
                logger.success("   ⭐ Excellente performance!")
            elif r2_test >= 0.90:
                logger.info("   👍 Très bonne performance!")
            elif r2_test >= 0.80:
                logger.warning("   ⚠️  Performance acceptable")
            else:
                logger.error("   ❌ Performance à améliorer")
    
    # 4. Prédictions
    predictions_path = Path("data/processed/test_predictions.csv")
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        logger.info("\n🔮 PRÉDICTIONS:")
        logger.info(f"   • Nombre de prédictions: {len(predictions)}")
        logger.info(f"   • Valeur moyenne prédite: {predictions.iloc[:, 0].mean():.2f}")
        logger.info(f"   • Écart-type: {predictions.iloc[:, 0].std():.2f}")
    
    # 5. Visualisations
    figures_dir = Path("reports/figures")
    if figures_dir.exists():
        plots = list(figures_dir.glob("*.png"))
        logger.info("\n📊 VISUALISATIONS GÉNÉRÉES:")
        for plot in plots:
            logger.info(f"   • {plot.name}")
    
    # 6. Fichiers générés
    logger.info("\n📁 FICHIERS GÉNÉRÉS:")
    files = [
        "data/processed/dataset.csv",
        "data/processed/features.csv",
        "data/processed/labels.csv",
        "models/model.pkl",
        "data/processed/test_predictions.csv"
    ]
    for file_path in files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size / 1024
            logger.info(f"   ✓ {file_path} ({size:.1f} KB)")
        else:
            logger.warning(f"   ✗ {file_path} (manquant)")
    
    logger.info("\n" + "="*80)
    logger.success("✨ PIPELINE COMPLET EXÉCUTÉ AVEC SUCCÈS!")
    logger.info("="*80)
    logger.info("\n💡 PROCHAINES ÉTAPES:")
    logger.info("   • Consulter les visualisations dans: reports/figures/")
    logger.info("   • Explorer le notebook: notebooks/ProjetIA_new.ipynb")
    logger.info("   • Utiliser le modèle entraîné pour de nouvelles prédictions")
    logger.info("\n")


def main():
    """Fonction principale."""
    logger.info("="*80)
    logger.info("🚀 LANCEMENT DU PIPELINE NATALITY-IA")
    logger.info("="*80)
    logger.info("")
    
    # Vérifier si les données existent
    data_path = Path("data/raw/fertility_rate.csv")
    if not data_path.exists():
        logger.warning("⚠️  Aucune donnée trouvée. Génération de données d'exemple...")
        if not run_command(
            "python scripts/download_sample_data.py --sample --countries 50",
            "Génération des données d'exemple"
        ):
            logger.error("❌ Échec de la génération des données")
            return
    
    # Pipeline complet
    steps = [
        ("python -m ProjetNatalite.dataset", "1/5 Traitement des données"),
        ("python -m ProjetNatalite.features", "2/5 Génération des features"),
        ("python -m ProjetNatalite.modeling.train", "3/5 Entraînement du modèle"),
        ("python -m ProjetNatalite.modeling.predict", "4/5 Génération des prédictions"),
        ("python -m ProjetNatalite.plots", "5/5 Création des visualisations"),
    ]
    
    for command, description in steps:
        if not run_command(command, description):
            logger.error("❌ Pipeline interrompu")
            return
    
    logger.info("")
    # Afficher le résumé
    display_summary()


if __name__ == "__main__":
    main()
