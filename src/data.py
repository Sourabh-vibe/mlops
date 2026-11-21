"""Export and validate scikit-learn's bundled Iris dataset (no network needed)."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from src.schema import FEATURE_NAMES, TARGET_COLUMN, TARGET_NAMES, sha256_file, write_json


def prepare_data(output: Path) -> pd.DataFrame:
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=list(FEATURE_NAMES))
    frame[TARGET_COLUMN] = iris.target
    validate_data(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format="%.1f", lineterminator="\n")
    return frame


def validate_data(frame: pd.DataFrame) -> dict[str, Any]:
    expected_columns = [*FEATURE_NAMES, TARGET_COLUMN]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Expected columns in order: {expected_columns}")
    if len(frame) != 150:
        raise ValueError("Iris data must contain exactly 150 observations")
    if any(not pd.api.types.is_numeric_dtype(frame[name]) for name in expected_columns):
        raise ValueError("All columns must be numeric")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Data contains missing or non-finite values")
    if (frame.loc[:, list(FEATURE_NAMES)] <= 0).any().any():
        raise ValueError("Feature measurements must be positive centimeters")
    target = frame[TARGET_COLUMN]
    if not pd.api.types.is_integer_dtype(target) or set(target.unique()) != {0, 1, 2}:
        raise ValueError("Target must contain integer class IDs 0, 1, and 2")
    counts = target.value_counts().sort_index()
    if not (counts == 50).all():
        raise ValueError("Expected exactly 50 observations for each species")
    # Iris contains a legitimate repeated measurement; do not drop duplicate rows.
    return {
        "valid": True,
        "rows": len(frame),
        "feature_names": list(FEATURE_NAMES),
        "class_counts": {TARGET_NAMES[int(key)]: int(value) for key, value in counts.items()},
    }


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_data(frame)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, default=Path("data/raw/iris.csv"))
    validate = commands.add_parser("validate")
    validate.add_argument("--data", type=Path, default=Path("data/raw/iris.csv"))
    validate.add_argument("--report", type=Path, default=Path("reports/data_validation.json"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_data(args.output)
    else:
        frame = load_data(args.data)
        report = validate_data(frame)
        report["dataset_sha256"] = sha256_file(args.data)
        write_json(args.report, report)


if __name__ == "__main__":
    main()
