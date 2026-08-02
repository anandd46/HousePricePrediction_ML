"""
visualization.py
-----------------
Generates and saves all visualizations for the House Price Prediction
project as PNG files inside the project folder.

Plots produced:
    1. price_distribution.png    - Distribution of house sale prices
    2. correlation_heatmap.png   - Correlation heatmap of numeric features
    3. feature_importance.png    - Feature importance from the best tree model
    4. prediction_vs_actual.png  - Predicted vs actual prices (best model)
    5. residual_plot.png         - Residuals vs predicted values
    6. error_distribution.png    - Histogram of prediction errors
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so plots can be saved headlessly
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid") if "seaborn-v0_8-darkgrid" in plt.style.available else None


def plot_price_distribution(df: pd.DataFrame, target_column: str = "SalePrice",
                             save_path: str = "price_distribution.png") -> None:
    """Plot and save the distribution of the target variable (house prices)."""
    plt.figure(figsize=(10, 6))
    plt.hist(df[target_column], bins=40, color="#4C72B0", edgecolor="black", alpha=0.8)
    plt.title("Distribution of House Sale Prices", fontsize=14, fontweight="bold")
    plt.xlabel("Sale Price ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = "correlation_heatmap.png") -> None:
    """Plot and save a correlation heatmap of numeric features using matplotlib only."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(14, 12))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    plt.title("Correlation Heatmap of Numeric Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_feature_importance(model, feature_names: List[str],
                             model_name: str = "Model",
                             save_path: str = "feature_importance.png") -> None:
    """
    Plot and save feature importance for a tree-based model
    (Random Forest or XGBoost). Skips gracefully if unsupported.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"[Visualization] Skipped feature importance (not supported by {model_name}).")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features

    plt.figure(figsize=(10, 8))
    plt.barh(
        [feature_names[i] for i in indices][::-1],
        [importances[i] for i in indices][::-1],
        color="#55A868",
    )
    plt.title(f"Top 15 Feature Importances ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_prediction_vs_actual(y_test: pd.Series, y_pred: np.ndarray,
                               model_name: str = "Model",
                               save_path: str = "prediction_vs_actual.png") -> None:
    """Plot and save predicted vs actual values with a perfect-prediction reference line."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color="#4C72B0", edgecolor="k", linewidth=0.3)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    plt.title(f"Predicted vs Actual House Prices ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_residuals(y_test: pd.Series, y_pred: np.ndarray,
                    model_name: str = "Model",
                    save_path: str = "residual_plot.png") -> None:
    """Plot and save residuals (actual - predicted) against predicted values."""
    residuals = y_test.values - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color="#C44E52", edgecolor="k", linewidth=0.3)
    plt.axhline(y=0, color="black", linestyle="--", linewidth=2)
    plt.title(f"Residual Plot ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def plot_error_distribution(y_test: pd.Series, y_pred: np.ndarray,
                             model_name: str = "Model",
                             save_path: str = "error_distribution.png") -> None:
    """Plot and save a histogram of prediction errors."""
    errors = y_test.values - y_pred

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=40, color="#8172B2", edgecolor="black", alpha=0.8)
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
    plt.title(f"Prediction Error Distribution ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Prediction Error ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


def generate_all_visualizations(
    df: pd.DataFrame,
    best_model,
    feature_names: List[str],
    y_test: pd.Series,
    y_pred: np.ndarray,
    best_model_name: str,
) -> None:
    """
    Convenience function to generate and save every visualization in one call.

    Args:
        df: Preprocessed DataFrame (for distribution/correlation plots).
        best_model: The best-performing fitted model.
        feature_names: List of feature column names.
        y_test: True test target values.
        y_pred: Best model's predictions on the test set.
        best_model_name: Name of the best model (used in plot titles).
    """
    print("\n[Visualization] Generating all plots...")
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_importance(best_model, feature_names, best_model_name)
    plot_prediction_vs_actual(y_test, y_pred, best_model_name)
    plot_residuals(y_test, y_pred, best_model_name)
    plot_error_distribution(y_test, y_pred, best_model_name)
    print("[Visualization] All plots generated successfully.\n")
