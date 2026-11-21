import json
import shutil

import pytest
from fastapi.testclient import TestClient

from api.app import PredictRequest, create_app

EXAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (EXAMPLE, "setosa"),
        (
            {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
            "versicolor",
        ),
        (
            {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2},
            "virginica",
        ),
    ],
)
def test_predict_all_species_and_probabilities(client, features, expected):
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    result = response.json()
    assert result["species"] == expected
    assert result["class_id"] == ["setosa", "versicolor", "virginica"].index(expected)
    assert set(result["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["confidence"] == max(result["probabilities"].values())
    assert len(result["model_version"]) == 12


def test_request_key_order_does_not_change_features(client):
    first = client.post("/predict", json=EXAMPLE).json()
    second = client.post("/predict", json=dict(reversed(list(EXAMPLE.items())))).json()
    assert first == second


@pytest.mark.parametrize("value", [-1, 0, 1e100, "5.1", "invalid", None, True])
def test_invalid_feature_returns_422(client, value):
    response = client.post("/predict", json={**EXAMPLE, "sepal_length": value})
    assert response.status_code == 422


def test_missing_feature_returns_422(client):
    response = client.post("/predict", json={"sepal_length": 5.1})
    assert response.status_code == 422


def test_unknown_feature_returns_422(client):
    assert client.post("/predict", json={**EXAMPLE, "typo": 1}).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurements_rejected(value):
    # JSON disallows these literals; check the schema directly without non-standard JSON.
    with pytest.raises(ValueError):
        PredictRequest(**{**EXAMPLE, "sepal_length": value})


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_http_payload_returns_422(client, literal):
    payload = json.dumps(EXAMPLE).replace("5.1", literal)
    response = client.post(
        "/predict", content=payload, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_health_and_readiness(client):
    assert client.get("/healthz").json() == {"status": "alive"}
    assert client.get("/readyz").json()["status"] == "ready"


def test_missing_artifact_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError), TestClient(create_app(tmp_path)):
        pass


def test_corrupt_model_fails_startup(tmp_path, trained_project):
    shutil.copytree(trained_project["root"] / "model", tmp_path / "model")
    with (tmp_path / "model" / "model.joblib").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"), TestClient(create_app(tmp_path / "model")):
        pass


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("feature_names", ["petal_width", "petal_length", "sepal_width", "sepal_length"], "schema"),
        ("target_names", ["virginica", "versicolor", "setosa"], "species mapping"),
        ("sklearn_version", "0.0.0", "versions must match"),
        ("model_version", "000000000000", "version does not match"),
    ],
)
def test_incompatible_manifest_fails_startup(tmp_path, trained_project, key, value, message):
    directory = tmp_path / "model"
    shutil.copytree(trained_project["root"] / "model", directory)
    path = directory / "metadata.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message), TestClient(create_app(directory)):
        pass


def test_unready_service_returns_503(client):
    client.app.state.bundle = None
    assert client.get("/readyz").status_code == 503
    assert client.post("/predict", json=EXAMPLE).status_code == 503


def test_inference_failure_is_sanitized(client, monkeypatch):
    def fail(_):
        raise RuntimeError("private implementation detail")

    monkeypatch.setattr(client.app.state.bundle.model, "predict_proba", fail)
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction unavailable"}
