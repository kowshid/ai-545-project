"""Register the best trained candidate as champion if it improves R2."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.utils import load_params, project_root


def _load_leaderboard(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError("metrics_comparison.json is empty")
    return rows


def main() -> None:
    params = load_params()
    root = project_root()
    model_path = root / params["paths"]["model"]
    comparison_path = root / "models" / "metrics_comparison.json"
    registry_dir = root / "registry"
    registry_dir.mkdir(exist_ok=True)
    champ_path = registry_dir / "champion.json"

    top = _load_leaderboard(comparison_path)[0]
    candidate_r2 = float(top["r2"])
    promote = True

    if champ_path.exists():
        with open(champ_path, "r", encoding="utf-8") as f:
            current = json.load(f)
        current_r2 = float(current["metrics"]["r2"])
        if candidate_r2 <= current_r2:
            promote = False
            print(f"[register] Keeping champion: current R2={current_r2:.4f}, candidate R2={candidate_r2:.4f}")

    if not promote:
        return

    champion = {
        "model_type": top["model_name"],
        "metrics": {"r2": top["r2"], "mae": top["mae"], "rmse": top["rmse"]},
        "model_path": str(model_path.relative_to(root)),
        "candidate_model_path": top["model_path"],
        "best_params": top.get("best_params", {}),
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": int(time.time()),
    }
    with open(champ_path, "w", encoding="utf-8") as f:
        json.dump(champion, f, indent=2)
    print(f"[register] New champion: {top['model_name']} (R2={candidate_r2:.4f})")


if __name__ == "__main__":
    main()