import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, confusion_matrix, classification_report
)

from loguru import logger


class LinearClassifier:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = None
        self.scaler = None

    def set_preprocessor(self, scaler):
        self.scaler = scaler

    def train(self, X_train, y_train, feature_names=None):
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        logger.success("Model trained successfully")

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(self, X, y, threshold=None):
        y_pred = self.predict(X)

        metrics = {
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
        }

        if threshold is not None:
            y_pred_bin = (y_pred >= threshold).astype(int)
            y_true_bin = (y >= threshold).astype(int)

            metrics.update({
                "accuracy": accuracy_score(y_true_bin, y_pred_bin),
                "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin).tolist(),
                "classification_report": classification_report(y_true_bin, y_pred_bin)
            })

        return metrics
