import numpy as np
import pandas as pd
import pytest

from src.data import load_data, prepare_data, validate_data
from src.schema import FEATURE_NAMES


def test_export_load_and_class_balance(tmp_path):
    path = tmp_path / "nested" / "iris.csv"
    original = prepare_data(path)
    loaded = load_data(path)
    pd.testing.assert_frame_equal(original, loaded)
    assert loaded.shape == (150, 5)
    assert list(loaded.columns[:4]) == list(FEATURE_NAMES)
    assert loaded["target"].value_counts().to_dict() == {0: 50, 1: 50, 2: 50}


def test_export_is_byte_reproducible(tmp_path):
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    prepare_data(first)
    prepare_data(second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_invalid_measurements_rejected(iris_frame, value):
    iris_frame.loc[0, "sepal_length"] = value
    with pytest.raises(ValueError):
        validate_data(iris_frame)


def test_wrong_schema_rejected(iris_frame):
    with pytest.raises(ValueError, match="columns"):
        validate_data(iris_frame.rename(columns={"sepal_length": "length"}))


def test_wrong_row_count_rejected(iris_frame):
    with pytest.raises(ValueError, match="150"):
        validate_data(iris_frame.iloc[:-1])


def test_wrong_class_balance_rejected(iris_frame):
    iris_frame.loc[0, "target"] = 1
    with pytest.raises(ValueError, match="50 observations"):
        validate_data(iris_frame)


def test_fractional_target_rejected(iris_frame):
    iris_frame["target"] = iris_frame["target"].astype(float) + 0.5
    with pytest.raises(ValueError, match="integer class IDs"):
        validate_data(iris_frame)


def test_non_numeric_column_rejected(iris_frame):
    iris_frame["sepal_length"] = "invalid"
    with pytest.raises(ValueError, match="numeric"):
        validate_data(iris_frame)


def test_legitimate_duplicate_is_preserved(iris_frame):
    assert np.count_nonzero(iris_frame.duplicated()) > 0
    assert validate_data(iris_frame)["valid"] is True
