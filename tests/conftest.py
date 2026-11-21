from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from src.config import PipelineConfig
from src.data import prepare_data
from src.train import train


@pytest.fixture
def iris_frame(tmp_path):
    return prepare_data(tmp_path / "iris.csv")


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("iris-project")
    data = root / "data" / "iris.csv"
    prepare_data(data)
    tracking_uri = f"sqlite:///{(root / 'tracking.db').as_posix()}"
    result = train(
        data,
        PipelineConfig(),
        root / "model",
        root / "metrics.json",
        tracking_uri=tracking_uri,
    )
    return {"root": root, "result": result, "tracking_uri": tracking_uri}


@pytest.fixture
def client(trained_project):
    model_dir: Path = trained_project["root"] / "model"
    with TestClient(create_app(model_dir)) as test_client:
        yield test_client
