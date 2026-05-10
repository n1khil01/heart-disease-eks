from __future__ import annotations
from typing import cast

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

DROP_COLUMNS = ["id", "dataset", "num"]

@dataclass
class Preprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler
    feature_columns: list[str]

def load_raw(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[df["chol"] == 0, "chol"] = np.nan

    df["target"] = (df["num"] > 0).astype(int)

    return df

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["target"]
    X = df.drop(columns=DROP_COLUMNS + ["target"])
    return X, y

def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        X,
        columns=CATEGORICAL_FEATURES,
        dummy_na=True,
        dtype=float,
    )

def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        ),
    )

def fit_preprocessor(X_train: pd.DataFrame) -> tuple[pd.DataFrame, Preprocessor]:
    X_train = X_train.copy()

    imputer = SimpleImputer(strategy="median")
    X_train[NUMERIC_FEATURES] = imputer.fit_transform(X_train[NUMERIC_FEATURES])

    X_train = X_train.fillna(0)

    scaler = StandardScaler()
    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])

    preprocessor = Preprocessor(
        imputer=imputer,
        scaler=scaler,
        feature_columns=list(X_train.columns),
    )
    return X_train, preprocessor

def transform(X: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
    X = X.copy()
    X[NUMERIC_FEATURES] = cast(np.ndarray, preprocessor.imputer.transform(X[NUMERIC_FEATURES]))
    X = X.fillna(0)
    X[NUMERIC_FEATURES] = preprocessor.scaler.transform(X[NUMERIC_FEATURES])

    X = X.reindex(columns=preprocessor.feature_columns, fill_value=0)
    return X

def prepare_data(
    csv_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Preprocessor]:
    df = load_raw(csv_path)
    df = clean(df)
    X, y = split_features_target(df)
    X = encode_categoricals(X)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    X_train, preprocessor = fit_preprocessor(X_train)
    X_test = transform(X_test, preprocessor)
    return X_train, X_test, y_train, y_test, preprocessor