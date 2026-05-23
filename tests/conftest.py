from __future__ import annotations

import pandas as pd
import pytest

from heart_disease_mlops.config import COLUMN_NAMES


@pytest.fixture()
def sample_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 1],
            [67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, "?", 2, 2],
            [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2, 0],
            [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, "?", 0],
        ],
        columns=COLUMN_NAMES,
    )
