from __future__ import annotations

import pandas as pd
import pytest

from heart_disease_mlops.config import FEATURE_COLUMNS
from heart_disease_mlops.features import build_model_pipeline, validate_feature_frame


def test_validate_feature_frame_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing feature columns"):
        validate_feature_frame(pd.DataFrame({"age": [63]}))


def test_model_pipeline_can_fit_small_dataset(sample_raw_dataframe: pd.DataFrame) -> None:
    dataframe = sample_raw_dataframe.copy()
    dataframe["target"] = (pd.to_numeric(dataframe["target"]) > 0).astype(int)
    dataframe["ca"] = pd.to_numeric(dataframe["ca"], errors="coerce")
    dataframe["thal"] = pd.to_numeric(dataframe["thal"], errors="coerce")

    model = build_model_pipeline("logistic_regression")
    model.fit(dataframe[FEATURE_COLUMNS], dataframe["target"])

    predictions = model.predict(dataframe[FEATURE_COLUMNS])
    assert len(predictions) == len(dataframe)
