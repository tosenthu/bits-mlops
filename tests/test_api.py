from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from heart_disease_mlops.api import create_app
from heart_disease_mlops.config import FEATURE_COLUMNS
from heart_disease_mlops.features import build_model_pipeline


def test_predict_endpoint_returns_probability(tmp_path: Path) -> None:
    training_df = pd.DataFrame(
        [
            [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],
            [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2],
            [56, 0, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, 0, 2],
            [57, 1, 0, 140, 192, 0, 1, 148, 0, 0.4, 1, 0, 1],
            [67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2],
            [62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2],
        ],
        columns=FEATURE_COLUMNS,
    )
    target = [1, 0, 0, 0, 1, 1]
    model = build_model_pipeline("logistic_regression")
    model.fit(training_df, target)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    app = create_app(model_path=model_path)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in {0, 1}
    assert 0 <= payload["heart_disease_probability"] <= 1
    assert 0 <= payload["confidence"] <= 1
