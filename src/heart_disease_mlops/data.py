"""Dataset acquisition and cleaning utilities."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from heart_disease_mlops.config import (
    COLUMN_NAMES,
    DATA_URL,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def download_raw_data(
    output_path: Path = RAW_DATA_PATH,
    url: str = DATA_URL,
    force: bool = False,
) -> Path:
    """Download the UCI Cleveland heart disease CSV if it is not present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    urlretrieve(url, output_path)
    return output_path


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw UCI file and assign descriptive column names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run scripts/download_data.py first."
        )

    return pd.read_csv(path, names=COLUMN_NAMES, na_values="?")


def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw columns to numeric types and binarize the target label."""
    cleaned_df = raw_df.copy()

    for column in COLUMN_NAMES:
        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")

    cleaned_df[TARGET_COLUMN] = (cleaned_df[TARGET_COLUMN] > 0).astype(int)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    return cleaned_df


def prepare_dataset(
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    download: bool = True,
    force_download: bool = False,
) -> Path:
    """Download, clean, and save the processed dataset."""
    if download:
        download_raw_data(raw_path, force=force_download)

    cleaned_df = clean_dataset(load_raw_data(raw_path))
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    return processed_path


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset, creating it from raw data when available."""
    if not path.exists():
        prepare_dataset(processed_path=path)
    return pd.read_csv(path)
