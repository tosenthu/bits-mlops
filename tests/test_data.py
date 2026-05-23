from __future__ import annotations

import pandas as pd

from heart_disease_mlops.config import TARGET_COLUMN
from heart_disease_mlops.data import clean_dataset


def test_clean_dataset_binarizes_target_and_converts_missing_values(
    sample_raw_dataframe: pd.DataFrame,
) -> None:
    cleaned = clean_dataset(sample_raw_dataframe)

    assert set(cleaned[TARGET_COLUMN].unique()) == {0, 1}
    assert cleaned["ca"].isna().sum() == 1
    assert cleaned["thal"].isna().sum() == 1
    assert cleaned["age"].dtype.kind in {"i", "f"}
