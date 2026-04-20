"""Register the freshly trained model as champion if it beats the current one."""
from __future__ import annotations
import json, shutil, time
from pathlib import Path

from src.utils import load_params, project_root

def main() -> None:
    params = load_params()
    root = project_root()

    metrics_path = root / params["paths"]["metrics"]
    model_path   = root / params["paths"]["model"]
    registry_dir = root / "registry"
    registry_dir.mkdir(exist_ok=True)
    champ_path   = registry_dir / "champion.json"

    with open(metrics_path) as f:
        new_metrics = json.load(f)

    promote = True
    if champ_path.exists():
        with open(champ_path) as f:
            current = json.load(f)
        # Higher R² wins; also accept ties
        if current["metrics"]["r2"] > new_metrics["r2"]:
            promote = False
            print(f"[register] Keeping champion (R²={current['metrics']['r2']:.4f} "
                  f"> candidate {new_metrics['r2']:.4f})")

    if promote:
        champion = {
            "model_type": params["model"]["type"],
            "metrics": new_metrics,
            "model_path": str(model_path.relative_to(root)),
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": int(time.time()),
        }
        with open(champ_path, "w") as f:
            json.dump(champion, f, indent=2)
        print(f"[register] New champion registered: R²={new_metrics['r2']:.4f}")

if __name__ == "__main__":
    main()