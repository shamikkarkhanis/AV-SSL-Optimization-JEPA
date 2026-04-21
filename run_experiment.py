from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jepa.config import deep_merge, load_and_resolve_config
from jepa.pipeline import run_experiment


logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train -> score -> evaluate for one experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", required=False, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--set", action="append", default=[], help="Inline JSON overrides for top-level config keys.")
    args = parser.parse_args()

    config = load_and_resolve_config(args.config)
    for override_json in args.set:
        config = deep_merge(config, json.loads(override_json))

    experiment_name = args.experiment_name or config.get("experiment", {}).get("name", "experiment")
    output_root = Path(config.get("experiment", {}).get("output_root", "experiments/runs"))
    run_dir = Path(args.run_dir) if args.run_dir else output_root / experiment_name
    run_experiment(config, run_dir)


if __name__ == "__main__":
    main()
