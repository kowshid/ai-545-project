"""
Promote the freshly-trained challenger to champion if (and only if) it
beats the current champion on the configured selection metric.

Reads:
    models/metrics.json       (written by train.py)
    registry/champion.json    (current champion, if any)

Writes:
    registry/champion.json    (only if the challenger wins)
"""
from __future__ import annotations

import json
import time

from src.utils import load_params, project_root

HIGHER_IS_BETTER = {"r2"}


def is_better(metric: str, candidate: float, current_best: float) -> bool:
    if metric in HIGHER_IS_BETTER:
        return candidate > current_best
    return candidate < current_best


def main() -> None:
    params = load_params()
    root = project_root()

    metrics_path = root / params["paths"]["metrics"]
    model_path = root / params["paths"]["model"]
    registry_dir = root / "registry"
    registry_dir.mkdir(exist_ok=True)
    champ_path = registry_dir / "champion.json"

    with open(metrics_path) as f:
        challenger = json.load(f)

    metric = challenger.get("selection_metric", "r2")
    challenger_score = challenger[metric]
    challenger_name = challenger.get("challenger_name", "unknown")
    challenger_type = challenger.get("challenger_type", "unknown")

    promote = True
    reason = "no existing champion"

    if champ_path.exists():
        with open(champ_path) as f:
            current = json.load(f)
        current_score = current["metrics"][metric]
        if is_better(metric, challenger_score, current_score):
            reason = (
                f"challenger {metric}={challenger_score:.4f} beats "
                f"champion {metric}={current_score:.4f}"
            )
        else:
            promote = False
            direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
            print(
                f"[register] Keeping current champion "
                f"({metric}={current_score:.4f}, {direction} is better) "
                f"— challenger only got {challenger_score:.4f}"
            )

    if promote:
        champion = {
            "challenger_name": challenger_name,
            "model_type": challenger_type,
            "selection_metric": metric,
            "metrics": {
                "r2": challenger["r2"],
                "mae": challenger["mae"],
                "rmse": challenger["rmse"],
            },
            "leaderboard": challenger.get("leaderboard", []),
            "model_path": str(model_path.relative_to(root)),
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": int(time.time()),
        }
        with open(champ_path, "w") as f:
            json.dump(champion, f, indent=2)
        print(
            f"[register] New champion: '{challenger_name}' "
            f"({metric}={challenger_score:.4f}) — {reason}"
        )


if __name__ == "__main__":
    main()
