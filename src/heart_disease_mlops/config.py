"""Shared project configuration."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

TARGET_COLUMN = "target"
FEATURE_COLUMNS = [column for column in COLUMN_NAMES if column != TARGET_COLUMN]

CATEGORICAL_COLUMNS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
