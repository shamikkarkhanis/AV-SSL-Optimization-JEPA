"""Compatibility wrapper for the legacy paper runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from jepa.config import load_and_resolve_config
from jepa.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper. Use run_experiment.py for the new public workflow.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", required=False, default=None)
    parser.add_argument("--hypothesis", default=None, help="Mapped to experiment.name for compatibility.")
    args = parser.parse_args()

    config = load_and_resolve_config(args.config)
    if args.hypothesis:
        config.setdefault("experiment", {})["name"] = args.hypothesis
    output_root = Path(config.get("experiment", {}).get("output_root", "experiments/runs"))
    run_dir = Path(args.run_dir) if args.run_dir else output_root / config["experiment"]["name"]
    run_experiment(config, run_dir)


if __name__ == "__main__":
    main()
