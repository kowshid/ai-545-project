"""
Lightweight tests so CI can verify the training pipeline and the
utility functions work without needing a GPU or external services.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.utils import demo_predict, load_params  # noqa: E402


def test_params_has_required_sections():
    p = load_params()
    # New shape: `training` instead of `model`
    for key in ("data", "features", "target", "training", "mlflow", "paths"):
        assert key in p, f"Missing section: {key}"

    feats = p["features"]
    for key in ("age", "bmi", "children", "sex", "smoker", "region"):
        assert key in feats, f"Missing feature: {key}"

    assert set(feats["sex"]) == {"female", "male"}
    assert set(feats["smoker"]) == {"yes", "no"}
    assert set(feats["region"]) == {"northeast", "northwest", "southeast", "southwest"}


def test_training_section_is_well_formed():
    p = load_params()
    train_cfg = p["training"]

    # Required keys for the multi-model trainer
    for key in ("test_size", "random_state", "selection_metric", "candidates"):
        assert key in train_cfg, f"Missing training key: {key}"

    assert train_cfg["selection_metric"] in {"r2", "mae", "rmse"}
    assert isinstance(train_cfg["candidates"], list) and len(train_cfg["candidates"]) >= 1

    # Each candidate must declare a name + a known model type
    from src.train import ESTIMATORS
    seen_names = set()
    for cand in train_cfg["candidates"]:
        assert "name" in cand and "type" in cand
        assert cand["name"] not in seen_names, f"Duplicate candidate name: {cand['name']}"
        seen_names.add(cand["name"])
        assert cand["type"] in ESTIMATORS, (
            f"Candidate '{cand['name']}' uses unknown model type '{cand['type']}'. "
            f"Allowed: {sorted(ESTIMATORS)}"
        )


def test_demo_predict_smoker_costs_more():
    base_row = {"age": 35, "sex": "male", "bmi": 28.0, "children": 1,
                "smoker": "no", "region": "northeast"}
    smoker_row = {**base_row, "smoker": "yes"}
    assert demo_predict(smoker_row) > demo_predict(base_row) + 10_000


def test_demo_predict_is_positive():
    row = {"age": 18, "sex": "female", "bmi": 20.0, "children": 0,
           "smoker": "no", "region": "southwest"}
    assert demo_predict(row) > 0


@pytest.mark.skipif(
    os.environ.get("SKIP_TRAIN_TEST") == "1",
    reason="Skip the slower training test when SKIP_TRAIN_TEST=1",
)
def test_first_candidate_can_train(tmp_path):
    """
    Train just the first candidate from params.yml end-to-end and
    confirm in-sample R² is reasonable. We don't run the whole
    multi-candidate pipeline here — that's what the CI `train` job does.
    """
    from src import train as train_mod
    from src.utils import load_dataset

    params = load_params()
    df = load_dataset(params)
    assert {"age", "sex", "bmi", "children", "smoker", "region", "charges"}.issubset(df.columns)
    assert len(df) > 1000

    cand = params["training"]["candidates"][0]
    pipeline = train_mod.build_pipeline(cand["type"], cand.get("params", {}))
    X = df.drop(columns=["charges"])
    y = df["charges"]
    pipeline.fit(X, y)

    from sklearn.metrics import r2_score
    preds = pipeline.predict(X)
    assert r2_score(y, preds) > 0.85
