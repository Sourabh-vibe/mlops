"""FastAPI entry point: uvicorn api.app:app --host 0.0.0.0 --port 8000."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.artifacts import ModelBundle, load_bundle
from src.schema import FEATURE_NAMES, TARGET_NAMES

logger = logging.getLogger(__name__)
Measurement = Annotated[
    float, Field(strict=True, gt=0, le=float(np.finfo(np.float32).max), allow_inf_nan=False)
]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Species = Literal["setosa", "versicolor", "virginica"]


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        },
    )

    sepal_length: Measurement = Field(description="Sepal length in centimeters")
    sepal_width: Measurement = Field(description="Sepal width in centimeters")
    petal_length: Measurement = Field(description="Petal length in centimeters")
    petal_width: Measurement = Field(description="Petal width in centimeters")


class PredictResponse(BaseModel):
    species: Species
    class_id: int = Field(ge=0, le=2)
    confidence: Probability
    probabilities: dict[Species, Probability]
    model_version: str


def get_bundle(request: Request) -> ModelBundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not ready"
        )
    return bundle


def create_app(model_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        directory = model_dir or Path(
            os.getenv(
                "IRIS_MODEL_DIR", str(Path(__file__).resolve().parents[1] / "models" / "iris")
            )
        )
        # Fail startup if artifacts are missing, corrupted, or incompatible.
        bundle = load_bundle(directory)
        application.state.bundle = bundle
        logger.info("Loaded Iris model %s", bundle.metadata.model_version)
        try:
            yield
        finally:
            application.state.bundle = None

    application = FastAPI(
        title="Iris Flower Classifier",
        description="Random Forest predictions from four measurements in centimeters.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Python's JSON parser accepts NaN/Infinity. Never echo non-finite inputs
        # into a JSON error response, where they would cause serialization to fail.
        errors = [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.get("/healthz", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/readyz", tags=["health"])
    def ready(bundle: Annotated[ModelBundle, Depends(get_bundle)]) -> dict[str, str]:
        return {"status": "ready", "model_version": bundle.metadata.model_version}

    @application.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(
        payload: PredictRequest, bundle: Annotated[ModelBundle, Depends(get_bundle)]
    ) -> PredictResponse:
        # Explicit order avoids relying on the order of keys in a JSON request.
        features = pd.DataFrame(
            [[getattr(payload, name) for name in FEATURE_NAMES]], columns=list(FEATURE_NAMES)
        )
        try:
            scores = bundle.model.predict_proba(features)[0]
        except Exception as exc:
            logger.exception("Iris inference failed")
            raise HTTPException(status_code=503, detail="Prediction unavailable") from exc
        winner = int(np.argmax(scores))
        class_id = int(bundle.model.classes_[winner])
        probabilities = {
            TARGET_NAMES[int(label)]: float(score)
            for label, score in zip(bundle.model.classes_, scores, strict=True)
        }
        return PredictResponse(
            species=TARGET_NAMES[class_id],
            class_id=class_id,
            confidence=float(scores[winner]),
            probabilities=probabilities,
            model_version=bundle.metadata.model_version,
        )

    return application


app = create_app()
