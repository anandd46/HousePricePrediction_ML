"""
predict.py
----------
Interactive command-line script to predict a house price using the
saved best model (saved_model.pkl).

Usage:
    python predict.py

The script will prompt for each required feature value, apply the exact
same encoding and scaling used during training, and print the predicted
sale price.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from model_utils import load_model

BUNDLE_PATH = "saved_model.pkl"

# Numeric features the user must supply, with friendly prompts and simple validation.
NUMERIC_FIELDS = [
    ("LotArea", "Lot Area (sq ft)", 3000, 20000),
    ("OverallQual", "Overall Quality (1-10)", 1, 10),
    ("OverallCond", "Overall Condition (1-10)", 1, 10),
    ("YearBuilt", "Year Built", 1900, 2026),
    ("YearRemodAdd", "Year Remodeled (same as YearBuilt if never remodeled)", 1900, 2026),
    ("TotalBsmtSF", "Total Basement Area (sq ft)", 0, 5000),
    ("GrLivArea", "Above Ground Living Area (sq ft)", 300, 6000),
    ("FullBath", "Full Bathrooms", 0, 5),
    ("HalfBath", "Half Bathrooms", 0, 3),
    ("BedroomAbvGr", "Bedrooms Above Ground", 0, 10),
    ("KitchenAbvGr", "Kitchens Above Ground", 1, 3),
    ("TotRmsAbvGrd", "Total Rooms Above Ground", 2, 15),
    ("Fireplaces", "Number of Fireplaces", 0, 4),
    ("GarageCars", "Garage Capacity (cars)", 0, 5),
    ("GarageArea", "Garage Area (sq ft)", 0, 1500),
    ("PoolArea", "Pool Area (sq ft, 0 if none)", 0, 1000),
    ("YrSold", "Year Sold", 2000, 2026),
    ("MoSold", "Month Sold (1-12)", 1, 12),
]

CATEGORICAL_FIELDS = [
    ("Neighborhood", "Neighborhood"),
    ("HouseStyle", "House Style"),
    ("GarageType", "Garage Type"),
    ("ExterCond", "Exterior Condition"),
]


def prompt_numeric(field_name: str, label: str, min_val: float, max_val: float) -> float:
    """
    Prompt the user for a numeric input with validation and retry on bad input.

    Args:
        field_name: Internal feature name (for error messages).
        label: Human-readable prompt label.
        min_val: Minimum acceptable value.
        max_val: Maximum acceptable value.

    Returns:
        A validated float value.
    """
    while True:
        raw = input(f"  {label} [{min_val}-{max_val}]: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print(f"    ⚠️  Invalid number for '{field_name}'. Please enter a numeric value.")
            continue

        if value < min_val or value > max_val:
            print(f"    ⚠️  Value should be between {min_val} and {max_val}. Try again.")
            continue

        return value


def prompt_categorical(field_name: str, label: str, encoder) -> str:
    """
    Prompt the user for a categorical input, showing valid known options.

    Args:
        field_name: Internal feature name.
        label: Human-readable prompt label.
        encoder: Fitted LabelEncoder for this column (used to show valid classes).

    Returns:
        A validated category string (falls back to "Unknown" if not recognized,
        which the encoder was trained to handle).
    """
    known_classes = list(encoder.classes_) if encoder is not None else []
    if known_classes:
        print(f"    (Known options: {', '.join(known_classes)})")

    while True:
        raw = input(f"  {label}: ").strip()
        if not raw:
            print("    ⚠️  This field cannot be empty.")
            continue
        if known_classes and raw not in known_classes:
            print(f"    ⚠️  '{raw}' was not seen during training. Using 'Unknown' instead.")
            return "Unknown" if "Unknown" in known_classes else known_classes[0]
        return raw


def collect_user_input(encoders: Dict[str, Any]) -> pd.DataFrame:
    """
    Collect all required feature values from the user via terminal prompts.

    Args:
        encoders: Dictionary of fitted LabelEncoders (one per categorical column).

    Returns:
        A single-row DataFrame with raw (unscaled, unencoded numeric-ready) values.
    """
    print("\n" + "=" * 60)
    print("🏠  HOUSE PRICE PREDICTION - Enter Property Details")
    print("=" * 60)

    values: Dict[str, Any] = {}

    print("\n--- Numeric Details ---")
    for field_name, label, min_val, max_val in NUMERIC_FIELDS:
        values[field_name] = prompt_numeric(field_name, label, min_val, max_val)

    print("\n--- Categorical Details ---")
    for field_name, label in CATEGORICAL_FIELDS:
        encoder = encoders.get(field_name)
        raw_value = prompt_categorical(field_name, label, encoder)
        values[field_name] = raw_value

    return pd.DataFrame([values])


def engineer_and_encode(input_df: pd.DataFrame, encoders: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply the same feature engineering and encoding used during training
    to a single raw input row.

    Args:
        input_df: Single-row DataFrame of raw user input.
        encoders: Dictionary of fitted LabelEncoders.

    Returns:
        A processed single-row DataFrame ready for scaling.
    """
    df = input_df.copy()

    # Feature engineering (must mirror preprocessing.engineer_features)
    df["HouseAge"] = max(0, df.loc[0, "YrSold"] - df.loc[0, "YearBuilt"])
    df["RemodAge"] = max(0, df.loc[0, "YrSold"] - df.loc[0, "YearRemodAdd"])
    df["TotalSF"] = df.loc[0, "GrLivArea"] + df.loc[0, "TotalBsmtSF"]
    df["TotalBath"] = df.loc[0, "FullBath"] + 0.5 * df.loc[0, "HalfBath"]
    df["IsRemodeled"] = int(df.loc[0, "YearBuilt"] != df.loc[0, "YearRemodAdd"])

    # Categorical encoding using the SAME fitted encoders from training
    for field_name, _ in CATEGORICAL_FIELDS:
        encoder = encoders.get(field_name)
        if encoder is not None:
            raw_value = df.loc[0, field_name]
            if raw_value in encoder.classes_:
                df[field_name] = encoder.transform([raw_value])[0]
            else:
                # Fallback: encode as the first known class to avoid a crash
                df[field_name] = encoder.transform([encoder.classes_[0]])[0]

    return df


def predict_price(bundle: Dict[str, Any], input_df: pd.DataFrame) -> float:
    """
    Predict the house price for a single processed input row.

    Args:
        bundle: Dictionary containing 'model', 'scaler', and 'feature_names'.
        input_df: Fully engineered and encoded single-row DataFrame.

    Returns:
        The predicted sale price as a float.
    """
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    # Ensure column order matches training exactly
    input_df = input_df[feature_names]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    return float(prediction)


def main() -> None:
    """Run the interactive prediction CLI."""
    try:
        bundle = load_model(BUNDLE_PATH)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    encoders = bundle["encoders"]
    model_name = bundle.get("model_name", "Best Model")

    print(f"[Info] Loaded best model: {model_name}")

    try:
        raw_input_df = collect_user_input(encoders)
        processed_df = engineer_and_encode(raw_input_df, encoders)
        predicted_price = predict_price(bundle, processed_df)

        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"  Model Used:       {model_name}")
        print(f"  Predicted Price:  ${predicted_price:,.2f}")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Prediction cancelled by user.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001 - top-level safety net for a CLI script
        print(f"\n❌ Unexpected error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
