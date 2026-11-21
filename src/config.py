"""Validated pipeline parameters; invalid configurations fail before training."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SplitConfig(StrictConfig):
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)


class ModelConfig(StrictConfig):
    n_estimators: int = Field(default=200, ge=1)
    max_depth: int | None = Field(default=5, ge=1)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)


class QualityConfig(StrictConfig):
    min_accuracy: float = Field(default=0.9, ge=0, le=1)


class TrackingConfig(StrictConfig):
    experiment_name: str = Field(default="iris-random-forest", min_length=1)


class PipelineConfig(StrictConfig):
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


def load_config(path: Path) -> PipelineConfig:
    with path.open(encoding="utf-8") as handle:
        return PipelineConfig.model_validate(yaml.safe_load(handle))
