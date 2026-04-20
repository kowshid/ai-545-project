"""
Lightweight tests so CI can verify the training pipeline and the
utility functions work without needing a GPU or external services.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.utils import demo_predict, load_params  # noqa: E402


def test_params_has_required_sections():
    p = load_params()
    for key in ("data", "features", "target", "model", "mlflow", "paths"):
        assert key in p, f"Missing section: {key}"

    feats = p["features"]
    for key in ("age", "bmi", "children", "sex", "smoker", "region"):
        assert key in feats, f"Missing feature: {key}"

    assert set(feats["sex"]) == {"female", "male"}
    assert set(feats["smoker"]) == {"yes", "no"}
    assert set(feats["region"]) == {"northeast", "northwest", "southeast", "southwest"}


def test_demo_predict_smoker_costs_more():
    base_row = {"age": 35, "sex": "male", "bmi": 28.0, "children": 1, "smoker": "no", "region": "northeast"}
    smoker_row = {**base_row, "smoker": "yes"}
    assert demo_predict(smoker_row) > demo_predict(base_row) + 10_000


def test_demo_predict_is_positive():
    row = {"age": 18, "sex": "female", "bmi": 20.0, "children": 0, "smoker": "no", "region": "southwest"}
    assert demo_predict(row) > 0


@pytest.mark.skipif(
    os.environ.get("SKIP_TRAIN_TEST") == "1",
    reason="Skip the slower training test when SKIP_TRAIN_TEST=1",
)
def test_full_training_pipeline(tmp_path, monkeypatch):
    """Train on the real CSV and assert R^2 is reasonable."""
    from src import train as train_mod
    from src.utils import load_dataset

    params = load_params()
    df = load_dataset(params)
    assert {"age", "sex", "bmi", "children", "smoker", "region", "charges"}.issubset(df.columns)
    assert len(df) > 1000

    pipeline = train_mod.build_pipeline(params["model"])
    X = df.drop(columns=["charges"])
    y = df["charges"]
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    # In-sample R^2 should be high for a RF on this size of data
    from sklearn.metrics import r2_score
    assert r2_score(y, preds) > 0.85
