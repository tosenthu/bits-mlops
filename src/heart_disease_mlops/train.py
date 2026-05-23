"""Command-line training entry point with MLflow tracking."""
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
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from heart_disease_mlops.config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
)
from heart_disease_mlops.data import load_processed_data, prepare_dataset
from heart_disease_mlops.modeling import (
    ModelRunResult,
    save_model,
    select_best_model,
    train_all_candidates,
    write_metrics_report,
)


def plot_model_comparison(results: list[ModelRunResult], output_path: Path) -> Path:
    """Create a compact model comparison chart."""
    rows = []
    for result in results:
        rows.append(
            {
                "model": result.model_name,
                "accuracy": result.test_metrics["accuracy"],
                "precision": result.test_metrics["precision"],
                "recall": result.test_metrics["recall"],
                "roc_auc": result.test_metrics["roc_auc"],
            }
        )
    metrics_df = pd.DataFrame(rows).melt(id_vars="model", var_name="metric", value_name="score")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=metrics_df, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Model comparison on held-out test set")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def log_run_to_mlflow(
    result: ModelRunResult,
    artifact_paths: list[Path],
    input_example: pd.DataFrame,
) -> None:
    """Log one candidate model run to MLflow."""
    with mlflow.start_run(run_name=result.model_name):
        mlflow.log_param("model_name", result.model_name)
        mlflow.log_param("cv_roc_auc", result.cv_roc_auc)
        for param_name, value in result.best_params.items():
            mlflow.log_param(param_name, value)

        mlflow.log_metric("cv_roc_auc", result.cv_roc_auc)
        for metric_name, value in result.test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        mlflow.sklearn.log_model(
            result.estimator,
            artifact_path="model",
            input_example=input_example,
            signature=ModelSignature(
                inputs=Schema([ColSpec("double", column) for column in FEATURE_COLUMNS]),
                outputs=Schema([ColSpec("long")]),
            ),
        )
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))


def train(
    data_path: Path = PROCESSED_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    metrics_path: Path = METRICS_PATH,
    tracking_uri: str = "mlruns",
    experiment_name: str = "heart-disease-uci",
    fast: bool = False,
) -> ModelRunResult:
    """Train, track, and persist the best heart disease classifier."""
    if not data_path.exists():
        prepare_dataset(processed_path=data_path)

    dataframe = load_processed_data(data_path)
    input_example = dataframe[FEATURE_COLUMNS].head(1).astype(float)
    results = train_all_candidates(dataframe, fast=fast)
    best_result = select_best_model(results)

    write_metrics_report(results, metrics_path)
    comparison_chart_path = plot_model_comparison(
        results,
        FIGURES_DIR / "model_comparison.png",
    )
    save_model(best_result, model_path, metadata_path)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    artifact_paths = [metrics_path, comparison_chart_path, metadata_path]
    for result in results:
        log_run_to_mlflow(result, artifact_paths, input_example)

    return best_result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train heart disease models.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=MODEL_METADATA_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--experiment-name", default="heart-disease-uci")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a reduced grid for CI/local smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    """Run model training from the command line."""
    args = parse_args()
    best_result = train(
        data_path=args.data_path,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        metrics_path=args.metrics_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        fast=args.fast,
    )
    print(f"Best model: {best_result.model_name}")
    print(f"Test metrics: {best_result.test_metrics}")


if __name__ == "__main__":
    main()
