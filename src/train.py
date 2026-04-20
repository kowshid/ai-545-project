"""
Train an insurance-charges regression model and log to MLflow.

Run:
    python -m src.train
or from the repo root:
    python src/train.py

Outputs:
    models/model.pkl       (the full sklearn Pipeline: preprocessor + estimator)
    models/metrics.json    (R^2, MAE, RMSE on the held-out test set)
    mlruns/                (MLflow tracking store)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow "python src/train.py" as well as "python -m src.train"
sys.path.append(str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils import load_dataset, load_params, project_root, save_model


CAT_COLS = ["sex", "smoker", "region"]
NUM_COLS = ["age", "bmi", "children"]


def build_pipeline(model_cfg: dict) -> Pipeline:
    """Preprocessing + RandomForest inside a single sklearn Pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    estimator = RandomForestRegressor(**model_cfg["params"])
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("model", estimator)]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mse)),
    }


def main() -> None:
    params = load_params()
    target = params["target"]
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]

    print("[train] Loading dataset…")
    df = load_dataset(params)
    print(f"[train] Dataset shape: {df.shape}")

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
    )

    # ---- MLflow ----
    tracking_uri = mlflow_cfg["tracking_uri"]
    if not tracking_uri.startswith(("http://", "https://", "file:")):
        # Relative path -> store inside the repo
        tracking_uri = f"file:{(project_root() / tracking_uri).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]):
        pipeline = build_pipeline(model_cfg)

        print("[train] Fitting pipeline…")
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        print(f"[train] Metrics: {metrics}")

        # Log params & metrics
        mlflow.log_params(model_cfg["params"])
        mlflow.log_param("model_type", model_cfg["type"])
        mlflow.log_param("test_size", model_cfg["test_size"])
        mlflow.log_metrics(metrics)

        # Log the model artifact
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Persist pickle for the Streamlit app / Docker image
        out = save_model(pipeline, params)
        print(f"[train] Saved pipeline to {out}")

        metrics_path = project_root() / params["paths"]["metrics"]
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[train] Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
