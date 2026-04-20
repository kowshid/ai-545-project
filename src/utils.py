"""
Shared utilities for loading params, data, and the model.
Used by both train.py and app.py so both sides stay consistent.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
import yaml


# ---------- Paths ----------

def project_root() -> Path:
    """Return the repository root (one level above src/)."""
    return Path(__file__).resolve().parent.parent


# ---------- Params ----------

def load_params(params_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load params.yml as a dictionary."""
    path = Path(params_path) if params_path else project_root() / "params.yml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- Data ----------

def download_data(url: str, local_path: str | os.PathLike) -> Path:
    """
    Download the insurance CSV to local_path if it is not already present.
    Returns the local Path.
    """
    local = Path(local_path)
    if not local.is_absolute():
        local = project_root() / local
    local.parent.mkdir(parents=True, exist_ok=True)

    if local.exists() and local.stat().st_size > 0:
        return local

    print(f"[data] Downloading dataset from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    local.write_bytes(resp.content)
    print(f"[data] Saved to {local}")
    return local


def load_dataset(params: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the insurance dataset as a DataFrame, downloading it first
    if needed. Columns: age, sex, bmi, children, smoker, region, charges.
    """
    data_cfg = params["data"]
    local = download_data(data_cfg["url"], data_cfg["local_path"])
    df = pd.read_csv(local)
    # Normalize string columns to lower case for consistency
    for col in ("sex", "smoker", "region"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    return df


# ---------- Model ----------

def model_path(params: Dict[str, Any]) -> Path:
    p = Path(params["paths"]["model"])
    if not p.is_absolute():
        p = project_root() / p
    return p


def load_model(params: Dict[str, Any]):
    """Return the trained pipeline, or None if the .pkl is missing."""
    mp = model_path(params)
    if not mp.exists():
        return None
    with open(mp, "rb") as f:
        return pickle.load(f)


def save_model(model, params: Dict[str, Any]) -> Path:
    mp = model_path(params)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "wb") as f:
        pickle.dump(model, f)
    return mp


# ---------- Demo fallback predictor ----------

def demo_predict(row: Dict[str, Any]) -> float:
    """
    A simple, interpretable formula used when model.pkl is not yet available.
    This matches the 'Demo Mode' described in the app.
    """
    age = float(row["age"])
    bmi = float(row["bmi"])
    children = float(row["children"])
    smoker = str(row["smoker"]).lower() == "yes"

    base = 2500.0
    charge = base + 260.0 * age + 310.0 * bmi + 475.0 * children
    if smoker:
        charge += 23000.0
        if bmi > 30:
            charge += 19000.0
    return float(max(charge, 1000.0))
