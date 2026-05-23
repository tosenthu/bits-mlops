from __future__ import annotations

import pandas as pd

from heart_disease_mlops.config import FEATURE_COLUMNS
from heart_disease_mlops.modeling import train_all_candidates


def test_train_all_candidates_fast_returns_two_models() -> None:
    rows = [
        [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 1],
        [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2, 0],
        [56, 0, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, 0, 2, 0],
        [57, 1, 0, 140, 192, 0, 1, 148, 0, 0.4, 1, 0, 1, 0],
        [67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2, 1],
        [62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2, 1],
        [52, 1, 1, 172, 199, 1, 1, 162, 0, 0.5, 2, 0, 2, 0],
        [58, 1, 2, 112, 230, 0, 0, 165, 0, 2.5, 1, 1, 2, 1],
        [44, 1, 1, 120, 263, 0, 1, 173, 0, 0.0, 2, 0, 2, 0],
        [60, 1, 0, 130, 206, 0, 0, 132, 1, 2.4, 1, 2, 3, 1],
        [54, 0, 2, 135, 304, 1, 1, 170, 0, 0.0, 2, 0, 2, 0],
        [64, 1, 3, 110, 211, 0, 0, 144, 1, 1.8, 1, 0, 2, 1],
    ]
    dataframe = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["target"])

    results = train_all_candidates(dataframe, fast=True)

    assert {result.model_name for result in results} == {"logistic_regression", "random_forest"}
    assert all(0 <= result.test_metrics["roc_auc"] <= 1 for result in results)
