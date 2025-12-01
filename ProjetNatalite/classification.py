"""
Classification module using Linear Regression and pre-trained models.

This module provides functionality for:
- Loading pre-trained models from pickle files
- Training linear regression models for classification/regression tasks
- Making predictions on new data
- Evaluating model performance
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from loguru import logger
import joblib

from ProjetNatalite.config import MODELS_DIR


class LinearClassifier:
    """
    Linear Regression based classifier for regression and classification tasks.
    
    Supports loading pre-trained models and training new ones.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Optional path to a pre-trained model pickle file
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path or MODELS_DIR / "model.pkl"
        self.feature_names = None
        
        if model_path and model_path.exists():
            self.load_model(model_path)

    def load_model(self, model_path: Path) -> None:
        """
        Load a pre-trained model from pickle file.
        
        Args:
            model_path: Path to the pickle model file
        """
        logger.info(f"Loading model from: {model_path}")
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Handle both direct model objects and dict-based models
            if isinstance(model_data, dict):
                self.model = model_data.get("model")
                self.scaler = model_data.get("scaler")
                self.feature_names = model_data.get("feature_names")
                logger.info(f"Loaded model dict with keys: {list(model_data.keys())}")
            else:
                self.model = model_data
                logger.info("Loaded model object directly")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def save_model(self, model_path: Optional[Path] = None) -> None:
        """
        Save the trained model to pickle file.
        
        Args:
            model_path: Path where to save the model
        """
        save_path = model_path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        
        logger.info(f"Saving model to: {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)
        logger.success(f"Model saved successfully")

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        random_state: int = 42,
        normalize: bool = True,
    ) -> dict:
        """
        Train a linear regression model.
        
        Args:
            X: Feature matrix (samples × features)
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            normalize: Whether to standardize features
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training...")
        
        # Store feature names if X is DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalize features if requested
        if normalize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        metrics = {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }
        
        logger.success("Model training complete")
        logger.info(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
        
        return metrics

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Apply scaling if it was used during training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        threshold: Optional[float] = None,
    ) -> dict:
        """
        Evaluate model on provided data.
        
        Args:
            X: Feature matrix
            y: True targets
            threshold: Optional threshold for classification (binary classification)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Regression metrics
        metrics = {
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
        }
        
        # Classification metrics if threshold provided
        if threshold is not None:
            y_pred_binary = (y_pred >= threshold).astype(int)
            y_binary = (y >= threshold).astype(int)
            
            metrics["accuracy"] = accuracy_score(y_binary, y_pred_binary)
            metrics["confusion_matrix"] = confusion_matrix(y_binary, y_pred_binary).tolist()
            
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"\nClassification Report:\n{classification_report(y_binary, y_pred_binary)}")
        else:
            logger.info(f"R² Score: {metrics['r2']:.4f}")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
        
        return metrics

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature coefficients (importance) from the linear model.
        
        Returns:
            DataFrame with feature names and their coefficients
        """
        if self.model is None:
            logger.warning("Model not trained or loaded yet")
            return None
        
        if self.feature_names is None:
            logger.warning("Feature names not available")
            return None
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_,
            "abs_coefficient": np.abs(self.model.coef_),
        }).sort_values("abs_coefficient", ascending=False)
        
        return importance_df


def train_classifier(
    features_path: Path,
    labels_path: Path,
    output_model_path: Optional[Path] = None,
) -> LinearClassifier:
    """
    Train a linear classifier from CSV files.
    
    Args:
        features_path: Path to features CSV file
        labels_path: Path to labels CSV file
        output_model_path: Path to save the trained model
        
    Returns:
        Trained LinearClassifier instance
    """
    logger.info(f"Loading features from: {features_path}")
    logger.info(f"Loading labels from: {labels_path}")
    
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).iloc[:, 0]  # Get first column as target
    
    classifier = LinearClassifier()
    classifier.train(X, y)
    
    if output_model_path:
        classifier.save_model(output_model_path)
    else:
        classifier.save_model()
    
    return classifier


def predict_with_model(
    model_path: Path,
    features_path: Path,
    output_predictions_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Load a model and make predictions on new data.
    
    Args:
        model_path: Path to the saved model pickle file
        features_path: Path to features CSV file
        output_predictions_path: Optional path to save predictions
        
    Returns:
        Array of predictions
    """
    logger.info(f"Loading model from: {model_path}")
    classifier = LinearClassifier(model_path)
    
    logger.info(f"Loading test features from: {features_path}")
    X_test = pd.read_csv(features_path)
    
    predictions = classifier.predict(X_test)
    
    if output_predictions_path:
        output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(predictions, columns=["prediction"]).to_csv(
            output_predictions_path, index=False
        )
        logger.success(f"Predictions saved to: {output_predictions_path}")
    
    return predictions


if __name__ == "__main__":
    # Example usage
    logger.info("Classification module loaded successfully")
