import json
from unittest.mock import patch

import numpy as np
import pytest
from mlflow import MlflowClient
from pydantic import ValidationError

from src.artifacts import load_bundle
from src.config import ModelConfig, PipelineConfig, load_config
from src.data import prepare_data
from src.schema import FEATURE_NAMES
from src.train import evaluate_model, fit_model, split_data, train


def test_training_uses_disjoint_stratified_holdout(iris_frame):
    x_train, x_test, y_train, y_test = split_data(iris_frame, PipelineConfig())
    assert len(x_train) == 120 and len(x_test) == 30
    assert set(x_train.index).isdisjoint(x_test.index)
    assert y_train.value_counts().to_dict() == {0: 40, 1: 40, 2: 40}
    assert y_test.value_counts().to_dict() == {0: 10, 1: 10, 2: 10}


def test_fit_is_reproducible_and_metrics_are_bounded(iris_frame):
    config = PipelineConfig()
    x_train, x_test, y_train, y_test = split_data(iris_frame, config)
    first = fit_model(x_train, y_train, config)
    second = fit_model(x_train, y_train, config)
    assert first.n_estimators == 200 and first.max_depth == 5
    np.testing.assert_array_equal(first.predict_proba(x_test), second.predict_proba(x_test))
    metrics = evaluate_model(first, x_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert metrics["accuracy"] >= config.quality.min_accuracy


@pytest.mark.integration
def test_training_logs_mlflow_model_and_serving_bundle(trained_project):
    result = trained_project["result"]
    root = trained_project["root"]
    bundle = load_bundle(root / "model")
    assert bundle.metadata.feature_names == list(FEATURE_NAMES)
    assert json.loads((root / "metrics.json").read_text()) == result["metrics"]
    client = MlflowClient(tracking_uri=trained_project["tracking_uri"])
    run = client.get_run(bundle.metadata.mlflow_run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["n_estimators"] == "200"
    assert run.data.params["max_depth"] == "5"
    assert run.data.params["random_state"] == "42"
    assert run.data.metrics == pytest.approx(result["metrics"])
    assert run.data.tags["quality_gate"] == "passed"
    # Download the MLflow model to prove it was persisted, not just referenced.
    import mlflow.sklearn

    logged_model = mlflow.sklearn.load_model(bundle.metadata.mlflow_model_uri)
    assert logged_model.n_features_in_ == 4


@pytest.mark.integration
def test_failed_quality_gate_preserves_existing_artifacts(tmp_path):
    data = tmp_path / "iris.csv"
    prepare_data(data)
    output = tmp_path / "model"
    output.mkdir()
    (output / "model.joblib").write_bytes(b"previous approved model")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"previous": true}')
    uri = f"sqlite:///{(tmp_path / 'failed.db').as_posix()}"
    with patch(
        "src.train.evaluate_model",
        return_value=dict.fromkeys(["accuracy", "precision", "recall", "f1"], 0.0),
    ):
        with pytest.raises(ValueError, match="model not published"):
            train(data, PipelineConfig(), output, metrics_path, tracking_uri=uri)
    assert (output / "model.joblib").read_bytes() == b"previous approved model"
    assert json.loads(metrics_path.read_text()) == {"previous": True}
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("iris-random-forest")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags["quality_gate"] == "failed"


@pytest.mark.parametrize(
    "parameters", [{"n_estimators": 0}, {"max_depth": -1}, {"random_state": -1}, {"unexpected": 1}]
)
def test_invalid_model_parameters_rejected(parameters):
    with pytest.raises(ValidationError):
        ModelConfig(**parameters)


def test_config_loads_yaml(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text("model:\n  n_estimators: 12\n  max_depth: null\n", encoding="utf-8")
    config = load_config(path)
    assert config.model.n_estimators == 12 and config.model.max_depth is None
