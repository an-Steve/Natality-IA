import pandas as pd
from sklearn.cluster import KMeans
from loguru import logger


class KMeansModel:

    def __init__(self, n_clusters=4, random_state=42):
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state
        )
        self.scaler = None

    def set_preprocessor(self, scaler):
        self.scaler = scaler

    def train(self, X):
        self.model.fit(X)
        logger.success("K-Means fitted successfully")

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def get_cluster_centers(self):
        return self.model.cluster_centers_
