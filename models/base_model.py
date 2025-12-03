import pickle
from loguru import logger

class BaseModel:
    def __init__(self, model, name: str):
        self.model = model
        self.name = name

    def train(self, X, y):
        logger.info(f"Entraînement du modèle {self.name}...")
        self.model.fit(X, y)

    def predict(self, X):
        logger.info(f"Prédictions avec le modèle {self.name}...")
        return self.model.predict(X)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.success(f"Modèle {self.name} sauvegardé dans {path}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
