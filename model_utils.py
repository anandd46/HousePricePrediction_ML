"""
model_utils.py
---------------
Utility functions for training, evaluating, comparing, and persisting
machine learning models for the House Price Prediction project.

Models covered:
    - Linear Regression
    - Random Forest Regressor
    - XGBoost Regressor
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def get_models() -> Dict[str, Any]:
    """
    Instantiate all models to be trained, with reasonable default hyperparameters.

    Returns:
        Dictionary mapping model name -> unfitted model instance.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_single_model(
    model: Any,
    X_train: np.ndarray,
    y_train: pd.Series,
) -> Tuple[Any, float]:
    """
    Train a single model and measure training time.

    Args:
        model: An unfitted scikit-learn-compatible regressor.
        X_train: Training features.
        y_train: Training target values.

    Returns:
        Tuple of (fitted model, training time in seconds).
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a fitted model on the test set using standard regression metrics.

    Metrics computed:
        - MAE  (Mean Absolute Error)
        - MSE  (Mean Squared Error)
        - RMSE (Root Mean Squared Error)
        - R2   (Coefficient of Determination)
        - Prediction time (seconds)

    Args:
        model: A fitted regressor.
        X_test: Test features.
        y_test: True test target values.

    Returns:
        Dictionary of metric name -> value.
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Prediction Time (s)": prediction_time,
        "y_pred": y_pred,
    }


def train_and_evaluate_all(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Train and evaluate all models, producing a comparison table.

    Args:
        X_train: Scaled training features.
        X_test: Scaled testing features.
        y_train: Training target.
        y_test: Testing target.

    Returns:
        Tuple of:
            - fitted_models: dict of model name -> fitted model
            - comparison_df: DataFrame comparing all models on all metrics
            - predictions: dict of model name -> predicted values (for plots)
    """
    models = get_models()
    fitted_models: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}
    results = []

    for name, model in models.items():
        print(f"\n[Training] {name} ...")
        fitted_model, train_time = train_single_model(model, X_train, y_train)
        metrics = evaluate_model(fitted_model, X_test, y_test)

        predictions[name] = metrics.pop("y_pred")
        fitted_models[name] = fitted_model

        row = {"Algorithm": name, "Training Time (s)": train_time, **metrics}
        results.append(row)

        print(
            f"[Result] {name}: MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f} "
            f"| R2={metrics['R2']:.4f} | Train Time={train_time:.3f}s"
        )

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[
        ["Algorithm", "MAE", "MSE", "RMSE", "R2", "Training Time (s)", "Prediction Time (s)"]
    ]
    comparison_df = comparison_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

    return fitted_models, comparison_df, predictions


def get_best_model(fitted_models: Dict[str, Any], comparison_df: pd.DataFrame) -> Tuple[str, Any]:
    """
    Identify the best-performing model based on the highest R2 score.

    Args:
        fitted_models: Dictionary of fitted models.
        comparison_df: Comparison DataFrame produced by train_and_evaluate_all.

    Returns:
        Tuple of (best model name, best fitted model instance).
    """
    best_name = comparison_df.iloc[0]["Algorithm"]
    return best_name, fitted_models[best_name]


def save_model(model: Any, filepath: str) -> None:
    """
    Persist a fitted model to disk using joblib.

    Args:
        model: Fitted model object.
        filepath: Destination path (e.g. 'xgboost_model.pkl').
    """
    joblib.dump(model, filepath)
    print(f"[Saved] Model saved to '{filepath}'")


def load_model(filepath: str) -> Any:
    """
    Load a previously saved model from disk.

    Args:
        filepath: Path to the .pkl model file.

    Returns:
        The deserialized model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    try:
        return joblib.load(filepath)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model file '{filepath}' not found. Please run train_model.py first."
        ) from exc
