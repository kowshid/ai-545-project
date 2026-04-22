"""Train and compare multiple candidate models, then save artifacts."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.utils import load_dataset, load_params, project_root

CAT_COLS = ["sex", "smoker", "region"]
NUM_COLS = ["age", "bmi", "children"]


def base_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mse)),
    }


def build_candidates(random_state: int) -> dict[str, tuple[Pipeline, dict]]:
    candidates: dict[str, tuple[Pipeline, dict]] = {}

    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", base_preprocessor()),
            ("model", RandomForestRegressor(random_state=random_state)),
        ]
    )
    rf_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [8, 12],
        "model__min_samples_split": [2, 4],
    }
    candidates["RandomForestRegressor"] = (rf_pipe, rf_grid)

    dt_pipe = Pipeline(
        steps=[
            ("preprocessor", base_preprocessor()),
            ("model", DecisionTreeRegressor(random_state=random_state)),
        ]
    )
    dt_grid = {
        "model__max_depth": [6, 10, 14],
        "model__min_samples_split": [2, 5, 8],
        "model__min_samples_leaf": [1, 2, 4],
    }
    candidates["DecisionTreeRegressor"] = (dt_pipe, dt_grid)

    ab_pipe = Pipeline(
        steps=[
            ("preprocessor", base_preprocessor()),
            ("model", AdaBoostRegressor(random_state=random_state)),
        ]
    )
    ab_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.1, 0.3],
        "model__loss": ["linear", "square"],
    }
    candidates["AdaBoostRegressor"] = (ab_pipe, ab_grid)
    return candidates


def main() -> None:
    params = load_params()
    target = params["target"]
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]
    root = project_root()

    df = load_dataset(params)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_cfg["test_size"], random_state=model_cfg["random_state"]
    )

    tracking_uri = mlflow_cfg["tracking_uri"]
    if not tracking_uri.startswith(("http://", "https://", "file:")):
        tracking_uri = f"file:{(root / tracking_uri).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    out_dir = root / "models"
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best_name = None
    best_model = None
    best_r2 = float("-inf")

    for name, (pipe, grid) in build_candidates(model_cfg["random_state"]).items():
        with mlflow.start_run(run_name=f"{mlflow_cfg['run_name']}_{name}"):
            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                cv=3,
                scoring="r2",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_param("model_name", name)
            mlflow.log_param("test_size", model_cfg["test_size"])
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            model_file = cand_dir / f"{name}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

            entry = {"model_name": name, "model_path": str(model_file.relative_to(root))}
            entry.update(metrics)
            entry["best_params"] = search.best_params_
            results.append(entry)

            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_name = name
                best_model = model

    assert best_name is not None and best_model is not None
    metrics_path = out_dir / "metrics.json"
    comparison_path = out_dir / "metrics_comparison.json"
    best_path = out_dir / "model.pkl"

    leaderboard = sorted(results, key=lambda r: r["r2"], reverse=True)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        winner = next(row for row in leaderboard if row["model_name"] == best_name)
        json.dump(
            {"model_name": best_name, "r2": winner["r2"], "mae": winner["mae"], "rmse": winner["rmse"]},
            f,
            indent=2,
        )

    with open(best_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"[train] Winner: {best_name} (R2={best_r2:.4f})")
    print(f"[train] Saved: {comparison_path}, {metrics_path}, {best_path}")


if __name__ == "__main__":
    main()
