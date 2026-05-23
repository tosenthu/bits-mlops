"""Generate EDA figures for the assignment report."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".cache" / "matplotlib").resolve()))
(Path.cwd() / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_disease_mlops.config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
)
from heart_disease_mlops.data import load_processed_data, prepare_dataset


def plot_class_balance(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "class_balance.png"
    plt.figure(figsize=(6, 4))
    axis = sns.countplot(
        data=dataframe,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        palette="Set2",
        legend=False,
    )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["No disease", "Disease"])
    axis.set_title("Target class balance")
    axis.set_xlabel("Target")
    axis.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_correlation_heatmap(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "correlation_heatmap.png"
    plt.figure(figsize=(10, 8))
    correlation = dataframe[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(numeric_only=True)
    sns.heatmap(correlation, cmap="vlag", center=0, linewidths=0.3)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_feature_histograms(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "feature_histograms.png"
    dataframe[["age", "trestbps", "chol", "thalach", "oldpeak"]].hist(
        bins=20,
        figsize=(11, 7),
        color="#4c78a8",
        edgecolor="white",
    )
    plt.suptitle("Key numeric feature distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def generate_eda(
    output_dir: Path = FIGURES_DIR,
    data_path: Path = PROCESSED_DATA_PATH,
) -> list[Path]:
    """Create report-ready EDA figures."""
    if not data_path.exists():
        prepare_dataset(processed_path=data_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = load_processed_data(data_path)
    return [
        plot_class_balance(dataframe, output_dir),
        plot_correlation_heatmap(dataframe, output_dir),
        plot_feature_histograms(dataframe, output_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA figures.")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = generate_eda(output_dir=args.output_dir, data_path=args.data_path)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
