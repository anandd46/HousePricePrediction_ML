
"""
train_model.py
---------------
Main training script for the House Price Prediction project.

Running this script will:
    1. Preprocess the dataset (dataset.csv)
    2. Train Linear Regression, Random Forest, and XGBoost models
    3. Evaluate and compare all models
    4. Generate visualizations (saved as PNG files)
    5. Save all trained models as .pkl files
    6. Save the best-performing model as 'saved_model.pkl'

Usage:
    python train_model.py
"""

from __future__ import annotations

import sys

import joblib

from preprocessing import full_preprocessing_pipeline
from model_utils import train_and_evaluate_all, get_best_model, save_model
from visualization import generate_all_visualizations


MODEL_FILENAMES = {
    "Linear Regression": "linear_model.pkl",
    "Random Forest": "randomforest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}


def print_comparison_table(comparison_df) -> None:
    """Pretty-print the model comparison table to the console."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON TABLE")
    print("=" * 90)
    formatted = comparison_df.copy()
    formatted["MAE"] = formatted["MAE"].map(lambda x: f"${x:,.2f}")
    formatted["MSE"] = formatted["MSE"].map(lambda x: f"{x:,.2f}")
    formatted["RMSE"] = formatted["RMSE"].map(lambda x: f"${x:,.2f}")
    formatted["R2"] = formatted["R2"].map(lambda x: f"{x:.4f}")
    formatted["Training Time (s)"] = formatted["Training Time (s)"].map(lambda x: f"{x:.3f}s")
    formatted["Prediction Time (s)"] = formatted["Prediction Time (s)"].map(lambda x: f"{x:.4f}s")
    print(formatted.to_string(index=False))
    print("=" * 90 + "\n")


def main() -> None:
    """Run the complete training pipeline end-to-end."""
    try:
        # Step 1: Preprocess the data
        artifacts = full_preprocessing_pipeline(csv_path="dataset.csv")

        X_train_scaled = artifacts["X_train_scaled"]
        X_test_scaled = artifacts["X_test_scaled"]
        y_train = artifacts["y_train"]
        y_test = artifacts["y_test"]
        feature_names = artifacts["feature_names"]
        scaler = artifacts["scaler"]
        encoders = artifacts["encoders"]
        raw_df = artifacts["raw_df"]

        # Step 2 & 3: Train and evaluate all models
        fitted_models, comparison_df, predictions = train_and_evaluate_all(
            X_train_scaled, X_test_scaled, y_train, y_test
        )

        print_comparison_table(comparison_df)

        # Step 4: Identify the best model
        best_name, best_model = get_best_model(fitted_models, comparison_df)
        best_r2 = comparison_df.iloc[0]["R2"]
        print(f"[Best Model] {best_name} (R2 = {best_r2:.4f})\n")

        # Step 5: Generate visualizations using the best model's predictions
        generate_all_visualizations(
            df=raw_df,
            best_model=best_model,
            feature_names=feature_names,
            y_test=y_test,
            y_pred=predictions[best_name],
            best_model_name=best_name,
        )

        # Step 6: Save all individual models
        for name, model in fitted_models.items():
            save_model(model, MODEL_FILENAMES[name])

        # Step 7: Save the best model separately, plus the scaler/encoders/feature list
        # bundled together so predict.py can reconstruct the exact preprocessing.
        bundle = {
            "model": best_model,
            "model_name": best_name,
            "scaler": scaler,
            "encoders": encoders,
            "feature_names": feature_names,
        }
        joblib.dump(bundle, "saved_model.pkl")
        print("[Saved] Best model bundle saved to 'saved_model.pkl'")

        # Step 8: Save comparison table as CSV for reference
        comparison_df.to_csv("model_comparison.csv", index=False)
        print("[Saved] Comparison table saved to 'model_comparison.csv'")

        print("\n✅ Training pipeline completed successfully!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - top-level safety net for a CLI script
        print(f"\n❌ Unexpected error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
