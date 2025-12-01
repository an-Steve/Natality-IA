import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
from loguru import logger


class RandomForestModel:

    def __init__(self,
                 n_estimators=300,
                 max_depth=None,
                 max_features="sqrt",
                 random_state=42):

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=max_features,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1
        )

        self.scaler = None
        self.feature_names = None

    def set_preprocessor(self, scaler):
        self.scaler = scaler

    def train(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.model.fit(X, y)
        logger.success("Random Forest trained successfully")

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred)
        }

        return metrics

    def get_feature_importances(self):
        if self.feature_names is None:
            return None
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        return importances
