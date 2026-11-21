"""Versioned, integrity-checked model bundles. Load only artifacts you trust."""

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Literal

import joblib
import numpy as np
import sklearn
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from src.schema import FEATURE_NAMES, TARGET_NAMES, sha256_file, write_json

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    model_version: str = Field(pattern=r"^[0-9a-f]{12}$")
    model_sha256: Digest
    dataset_sha256: Digest
    sklearn_version: str
    feature_names: list[str]
    target_names: list[str]
    mlflow_run_id: str
    mlflow_model_uri: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ModelBundle:
    model: RandomForestClassifier
    metadata: ModelMetadata


def save_bundle(model: RandomForestClassifier, output: Path, **metadata: Any) -> ModelMetadata:
    output.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=output, suffix=".joblib", delete=False) as handle:
            temporary = Path(handle.name)
        joblib.dump(model, temporary, compress=3)
        digest = sha256_file(temporary)
        manifest = ModelMetadata(
            model_version=digest[:12],
            model_sha256=digest,
            sklearn_version=sklearn.__version__,
            feature_names=list(FEATURE_NAMES),
            target_names=list(TARGET_NAMES),
            **metadata,
        )
        os.replace(temporary, output / "model.joblib")
        # The manifest is written last. Do not update a running deployment in place.
        write_json(output / "metadata.json", manifest.model_dump())
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_bundle(directory: Path) -> ModelBundle:
    manifest = ModelMetadata.model_validate_json(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    if manifest.feature_names != list(FEATURE_NAMES):
        raise ValueError("Model feature schema does not match the API")
    if manifest.target_names != list(TARGET_NAMES):
        raise ValueError("Model species mapping does not match the API")
    if manifest.sklearn_version != sklearn.__version__:
        raise ValueError("Model and serving scikit-learn versions must match exactly")
    path = directory / "model.joblib"
    if sha256_file(path) != manifest.model_sha256:
        raise ValueError("Model checksum mismatch")
    if manifest.model_version != manifest.model_sha256[:12]:
        raise ValueError("Model version does not match checksum")
    # A checksum detects corruption; it does not make untrusted pickle files safe.
    model = joblib.load(path)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Artifact must contain a RandomForestClassifier")
    check_is_fitted(model)
    if list(model.feature_names_in_) != list(FEATURE_NAMES):
        raise ValueError("Fitted model feature order does not match the API")
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError("Fitted model must predict classes 0, 1, and 2")
    return ModelBundle(model=model, metadata=manifest)
