"""Training, evaluation, and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from heart_disease_mlops.config import FEATURE_COLUMNS, TARGET_COLUMN
from heart_disease_mlops.features import ModelName, build_model_pipeline


@dataclass(frozen=True)
class ModelRunResult:
    """Serializable summary of a trained model candidate."""

    model_name: str
    estimator: Pipeline
    best_params: dict[str, Any]
    cv_roc_auc: float
    test_metrics: dict[str, float]
    confusion_matrix: list[list[int]]


def get_param_grid(model_name: ModelName, fast: bool = False) -> dict[str, list[Any]]:
    """Return the parameter grid used for model selection."""
    if model_name == "logistic_regression":
        values = [1.0] if fast else [0.1, 1.0, 10.0]
        return {"classifier__C": values}

    if model_name == "random_forest":
        if fast:
            return {
                "classifier__n_estimators": [50],
                "classifier__max_depth": [None],
                "classifier__min_samples_leaf": [1],
            }
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_leaf": [1, 2],
        }

    raise ValueError(f"Unsupported model name: {model_name}")


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into model features and binary target."""
    return dataframe[FEATURE_COLUMNS], dataframe[TARGET_COLUMN].astype(int)


def evaluate_estimator(
    estimator: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], list[list[int]]]:
    """Evaluate a classifier with assignment-required metrics."""
    y_pred = estimator.predict(x_test)

    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(x_test)[:, 1]
    else:
        y_score = y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
    }
    matrix = confusion_matrix(y_test, y_pred).astype(int).tolist()
    return metrics, matrix


def train_candidate(
    model_name: ModelName,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
    fast: bool = False,
) -> ModelRunResult:
    """Train and tune one model candidate."""
    estimator = build_model_pipeline(model_name, random_state=random_state)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=get_param_grid(model_name, fast=fast),
        scoring="roc_auc",
        cv=3 if fast else 5,
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    test_metrics, matrix = evaluate_estimator(search.best_estimator_, x_test, y_test)

    return ModelRunResult(
        model_name=model_name,
        estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        cv_roc_auc=float(search.best_score_),
        test_metrics=test_metrics,
        confusion_matrix=matrix,
    )


def train_all_candidates(
    dataframe: pd.DataFrame,
    random_state: int = 42,
    fast: bool = False,
) -> list[ModelRunResult]:
    """Train all required model candidates and return their summaries."""
    x, y = split_features_target(dataframe)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    return [
        train_candidate(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            random_state=random_state,
            fast=fast,
        )
        for model_name in ("logistic_regression", "random_forest")
    ]


def select_best_model(results: list[ModelRunResult]) -> ModelRunResult:
    """Select the best candidate by held-out ROC-AUC, then F1."""
    if not results:
        raise ValueError("No model results were provided.")

    return max(
        results,
        key=lambda result: (result.test_metrics["roc_auc"], result.test_metrics["f1"]),
    )


def save_model(result: ModelRunResult, model_path: Path, metadata_path: Path) -> None:
    """Persist the final pipeline and model metadata."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.estimator, model_path)
    metadata = {
        "model_name": result.model_name,
        "feature_columns": FEATURE_COLUMNS,
        "best_params": result.best_params,
        "cv_roc_auc": result.cv_roc_auc,
        "test_metrics": result.test_metrics,
        "confusion_matrix": result.confusion_matrix,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_metrics_report(results: list[ModelRunResult], output_path: Path) -> None:
    """Write model comparison metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": [
            {
                "model_name": result.model_name,
                "best_params": result.best_params,
                "cv_roc_auc": result.cv_roc_auc,
                "test_metrics": result.test_metrics,
                "confusion_matrix": result.confusion_matrix,
            }
            for result in results
        ],
        "best_model": select_best_model(results).model_name,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def predict_probability(estimator: Pipeline, feature_df: pd.DataFrame) -> np.ndarray:
    """Return positive-class probability for model-serving code."""
    probabilities = estimator.predict_proba(feature_df)
    return probabilities[:, 1]
