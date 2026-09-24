"""
preprocessing.py
-----------------
Handles all data preprocessing steps for the House Price Prediction project:

    1. Loading the raw dataset
    2. Handling missing values
    3. Removing duplicate rows
    4. Outlier detection & removal (IQR method)
    5. Feature engineering (derived features)
    6. Encoding categorical columns
    7. Feature scaling
    8. Feature selection
    9. Train/test split

Every function is typed, documented, and safe to import from other scripts
(train_model.py, predict.py, house_price_prediction.py).
"""

from __future__ import annotations

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

TARGET_COLUMN = "SalePrice"


def load_dataset(csv_path: str = "dataset.csv") -> pd.DataFrame:
    """
    Load the raw house price dataset from a CSV file.

    Args:
        csv_path: Path to the dataset CSV file.

    Returns:
        A pandas DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Please place 'dataset.csv' "
            f"inside the project folder. See README.md for download instructions."
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("The dataset file is empty. Please provide a valid dataset.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from the DataFrame."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"[Preprocessing] Removed {before - after} duplicate rows.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategy:
        - Numeric columns: filled with the column median (robust to outliers).
        - Categorical columns: filled with the string "Unknown".

    Args:
        df: Input DataFrame that may contain NaNs.

    Returns:
        DataFrame with missing values imputed.
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    missing_after = df.isnull().sum().sum()
    print(f"[Preprocessing] Missing values before: {missing_before}, after: {missing_after}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified numeric columns using the IQR method.

    Args:
        df: Input DataFrame.
        columns: List of numeric column names to check for outliers.
        factor: IQR multiplier (1.5 is the standard "mild outlier" threshold).

    Returns:
        DataFrame with outlier rows removed.
    """
    df = df.copy()
    rows_before = len(df)

    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    rows_after = len(df)
    print(f"[Preprocessing] Removed {rows_before - rows_after} outlier rows.")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new, more predictive features from existing raw columns.

    New features:
        - HouseAge: Age of the house at time of sale.
        - RemodAge: Years since last remodel at time of sale.
        - TotalSF: Combined living + basement square footage.
        - TotalBath: Weighted total bathroom count (half bath = 0.5).
        - IsRemodeled: Binary flag if remodel year differs from build year.

    Args:
        df: Preprocessed DataFrame.

    Returns:
        DataFrame with additional engineered feature columns.
    """
    df = df.copy()

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)

    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)

    if {"GrLivArea", "TotalBsmtSF"}.issubset(df.columns):
        df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

    if {"FullBath", "HalfBath"}.issubset(df.columns):
        df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"]

    if {"YearBuilt", "YearRemodAdd"}.issubset(df.columns):
        df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    print(f"[Preprocessing] Feature engineering complete. New shape: {df.shape}")
    return df


def encode_categorical_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical (object-type) columns using Label Encoding.

    Args:
        df: DataFrame possibly containing categorical columns.

    Returns:
        A tuple of (encoded DataFrame, dict of fitted LabelEncoders per column).
        The encoders are returned so the exact same encoding can later be
        applied to new/unseen data during prediction.
    """
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    print(f"[Preprocessing] Encoded {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
    return df, encoders


def select_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into features (X) and target (y).

    Args:
        df: Fully preprocessed DataFrame.
        target_column: Name of the target column to predict.

    Returns:
        A tuple (X, y) where X is the feature DataFrame and y is the target Series.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale numeric features using StandardScaler (zero mean, unit variance).

    Fit is done ONLY on the training set to avoid data leakage, then applied
    to both train and test sets.

    Args:
        X_train: Training feature set.
        X_test: Testing feature set.

    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def full_preprocessing_pipeline(
    csv_path: str = "dataset.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Run the complete preprocessing pipeline end-to-end.

    Steps:
        load -> remove duplicates -> handle missing values -> engineer features
        -> remove outliers -> encode categoricals -> split X/y -> train/test split
        -> scale features

    Args:
        csv_path: Path to the raw dataset CSV.
        test_size: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.

    Returns:
        A dictionary containing all pipeline artifacts:
        {
            "X_train", "X_test", "y_train", "y_test",
            "X_train_scaled", "X_test_scaled",
            "scaler", "encoders", "feature_names", "raw_df"
        }
    """
    print("=" * 60)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_dataset(csv_path)
    print(f"[Preprocessing] Loaded dataset with shape: {df.shape}")

    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = engineer_features(df)

    # Only remove outliers on key numeric columns to avoid over-pruning the dataset
    outlier_columns = ["SalePrice", "GrLivArea", "LotArea", "TotalSF"]
    outlier_columns = [c for c in outlier_columns if c in df.columns]
    df = remove_outliers_iqr(df, outlier_columns)

    df, encoders = encode_categorical_columns(df)

    X, y = select_features(df, TARGET_COLUMN)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[Preprocessing] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("=" * 60)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
        "raw_df": df,
    }


if __name__ == "__main__":
    # Allows running this file standalone to sanity-check the pipeline
    artifacts = full_preprocessing_pipeline()
    print("\nFeature columns:", artifacts["feature_names"])
