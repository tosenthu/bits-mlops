"""Feature preprocessing and model pipeline construction."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease_mlops.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
)

ModelName = Literal["logistic_regression", "random_forest"]


def validate_feature_frame(feature_df: pd.DataFrame) -> None:
    """Raise a useful error when the model input is missing required features."""
    missing_columns = sorted(set(FEATURE_COLUMNS) - set(feature_df.columns))
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")


def build_preprocessor() -> ColumnTransformer:
    """Build a reusable preprocessing transformer for model training and serving."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(model_name: ModelName, random_state: int = 42) -> Pipeline:
    """Create a full preprocessing plus classifier pipeline."""
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=random_state,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
