"""
Train multiple candidate models, log each as a separate MLflow run,
pick the best one as the *challenger*, and save it as model.pkl.

Run:
    python -m src.train

Outputs:
    models/model.pkl       (best sklearn Pipeline by selection_metric)
    models/metrics.json    (metrics for the challenger + leaderboard of all candidates)
    mlruns/                (one MLflow run per candidate, plus a parent "training" run)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python src/train.py` as well as `python -m src.train`
sys.path.append(str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils import load_dataset, load_params, project_root, save_model

CAT_COLS = ["sex", "smoker", "region"]
NUM_COLS = ["age", "bmi", "children"]

# Whitelist of model classes the YAML is allowed to reference.
ESTIMATORS = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "LinearRegression": LinearRegression,
}

# Higher-is-better metrics. Anything else is treated as lower-is-better.
HIGHER_IS_BETTER = {"r2"}


def build_pipeline(model_type: str, model_params: dict) -> Pipeline:
    """Build a preprocessing + estimator pipeline for one candidate."""
    if model_type not in ESTIMATORS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Allowed: {sorted(ESTIMATORS)}"
        )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    estimator = ESTIMATORS[model_type](**(model_params or {}))
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mse)),
    }


def is_better(metric: str, candidate: float, current_best: float | None) -> bool:
    """Compare two scores honoring the metric's optimization direction."""
    if current_best is None:
        return True
    if metric in HIGHER_IS_BETTER:
        return candidate > current_best
    return candidate < current_best


def main() -> None:
    params = load_params()
    target = params["target"]
    train_cfg = params["training"]
    mlflow_cfg = params["mlflow"]
    selection_metric = train_cfg.get("selection_metric", "r2")

    candidates = train_cfg.get("candidates", [])
    if not candidates:
        raise SystemExit("No candidates configured under training.candidates in params.yml.")

    print(f"[train] {len(candidates)} candidate(s) to train")
    print(f"[train] Selection metric: {selection_metric} "
          f"({'higher' if selection_metric in HIGHER_IS_BETTER else 'lower'} is better)")

    # ---- Data ----
    print("[train] Loading dataset…")
    df = load_dataset(params)
    print(f"[train] Dataset shape: {df.shape}")

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
    )

    # ---- MLflow ----
    tracking_uri = mlflow_cfg["tracking_uri"]
    if not tracking_uri.startswith(("http://", "https://", "file:")):
        tracking_uri = f"file:{(project_root() / tracking_uri).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    leaderboard: list[dict] = []
    best_metrics: dict | None = None
    best_pipeline: Pipeline | None = None
    best_name: str | None = None
    best_type: str | None = None

    # Parent run that groups all candidates
    with mlflow.start_run(run_name="training_session") as parent:
        mlflow.log_param("n_candidates", len(candidates))
        mlflow.log_param("selection_metric", selection_metric)
        mlflow.log_param("test_size", train_cfg["test_size"])

        for cand in candidates:
            name = cand["name"]
            mtype = cand["type"]
            mparams = cand.get("params", {}) or {}

            with mlflow.start_run(run_name=name, nested=True):
                pipeline = build_pipeline(mtype, mparams)

                print(f"[train] Fitting {name} ({mtype})…")
                pipeline.fit(X_train, y_train)

                metrics = evaluate(pipeline, X_test, y_test)
                print(f"[train]   → {metrics}")

                mlflow.log_param("candidate_name", name)
                mlflow.log_param("model_type", mtype)
                if mparams:
                    mlflow.log_params(mparams)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipeline, artifact_path="model")

                row = {"name": name, "model_type": mtype, **metrics}
                leaderboard.append(row)

                cand_score = metrics[selection_metric]
                best_score = best_metrics[selection_metric] if best_metrics else None
                if is_better(selection_metric, cand_score, best_score):
                    best_metrics = metrics
                    best_pipeline = pipeline
                    best_name = name
                    best_type = mtype

        assert best_pipeline is not None and best_metrics is not None

        # Sort leaderboard for readability
        reverse = selection_metric in HIGHER_IS_BETTER
        leaderboard.sort(key=lambda r: r[selection_metric], reverse=reverse)

        print("\n[train] Leaderboard (sorted by {}):".format(selection_metric))
        for r in leaderboard:
            marker = " ← challenger" if r["name"] == best_name else ""
            print(f"  {r['name']:<20} r2={r['r2']:.4f}  mae={r['mae']:.2f}  rmse={r['rmse']:.2f}{marker}")

        # Tag the parent run with the winner so it's findable in the MLflow UI
        mlflow.set_tag("challenger_name", best_name)
        mlflow.set_tag("challenger_type", best_type)
        mlflow.log_metrics({f"challenger_{k}": v for k, v in best_metrics.items()})

        # ---- Persist the challenger as the deployable artifact ----
        out = save_model(best_pipeline, params)
        print(f"\n[train] Saved challenger '{best_name}' to {out}")

        metrics_path = project_root() / params["paths"]["metrics"]
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **best_metrics,  # r2, mae, rmse of the challenger
            "challenger_name": best_name,
            "challenger_type": best_type,
            "selection_metric": selection_metric,
            "leaderboard": leaderboard,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[train] Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
