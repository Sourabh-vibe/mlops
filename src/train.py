"""Train, evaluate, log to MLflow, and export a model after the quality gate passes."""

import argparse
import logging
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from src.artifacts import save_bundle
from src.config import PipelineConfig, load_config
from src.data import load_data
from src.schema import FEATURE_NAMES, TARGET_COLUMN, TARGET_NAMES, sha256_file, write_json

logger = logging.getLogger(__name__)


def split_data(frame: pd.DataFrame, config: PipelineConfig) -> tuple:
    return train_test_split(
        frame.loc[:, list(FEATURE_NAMES)],
        frame[TARGET_COLUMN],
        test_size=config.split.test_size,
        random_state=config.split.random_state,
        stratify=frame[TARGET_COLUMN],
    )


def fit_model(
    features: pd.DataFrame, target: pd.Series, config: PipelineConfig
) -> RandomForestClassifier:
    # One parallelism layer: increase service workers for deployment concurrency.
    model = RandomForestClassifier(**config.model.model_dump(), n_jobs=1)
    return model.fit(features, target)


def evaluate_model(
    model: RandomForestClassifier, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    predictions = model.predict(features)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, predictions, labels=[0, 1, 2], average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train(
    data_path: Path,
    config: PipelineConfig,
    output: Path,
    metrics_path: Path,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    frame = load_data(data_path)
    x_train, x_test, y_train, y_test = split_data(frame, config)
    dataset_digest = sha256_file(data_path)
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        uri = f"sqlite:///{(Path.cwd() / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config.tracking.experiment_name)
    with mlflow.start_run(run_name="random-forest") as run:
        mlflow.set_tags(
            {
                "dataset": "sklearn-iris",
                "dataset_sha256": dataset_digest,
                "git_commit": os.getenv("GITHUB_SHA", "local"),
                "metric_average": "macro",
            }
        )
        mlflow.log_params(
            {
                **config.model.model_dump(),
                "test_size": config.split.test_size,
                "split_random_state": config.split.random_state,
                "min_accuracy": config.quality.min_accuracy,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
            }
        )
        model = fit_model(x_train, y_train, config)
        metrics = evaluate_model(model, x_test, y_test)
        mlflow.log_metrics(metrics)
        predictions = model.predict(x_test)
        mlflow.log_dict(
            {
                "confusion_matrix": confusion_matrix(
                    y_test, predictions, labels=[0, 1, 2]
                ).tolist(),
                "classification_report": classification_report(
                    y_test,
                    predictions,
                    labels=[0, 1, 2],
                    target_names=list(TARGET_NAMES),
                    output_dict=True,
                    zero_division=0,
                ),
            },
            "evaluation.json",
        )
        if metrics["accuracy"] < config.quality.min_accuracy:
            mlflow.set_tag("quality_gate", "failed")
            raise ValueError(
                f"Accuracy {metrics['accuracy']:.4f} is below "
                f"required minimum {config.quality.min_accuracy:.4f}; model not published"
            )
        mlflow.set_tag("quality_gate", "passed")
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest",
            signature=infer_signature(x_train, model.predict(x_train)),
            input_example=x_train.head(3),
            pip_requirements=[
                f"{package}=={version(package)}"
                for package in ("scikit-learn", "numpy", "scipy", "pandas", "joblib")
            ],
        )
        run_id = run.info.run_id
        model_uri = info.model_uri
    # Export only after MLflow finishes successfully. A tracking outage fails training.
    manifest = save_bundle(
        model,
        output,
        dataset_sha256=dataset_digest,
        mlflow_run_id=run_id,
        mlflow_model_uri=model_uri,
        parameters=config.model_dump(),
    )
    write_json(metrics_path, metrics)
    logger.info("Model %s passed the quality gate: %s", manifest.model_version, metrics)
    return {"metrics": metrics, "metadata": manifest.model_dump()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/iris.csv"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/iris"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train(args.data, load_config(args.params), args.output, args.metrics)


if __name__ == "__main__":
    main()
