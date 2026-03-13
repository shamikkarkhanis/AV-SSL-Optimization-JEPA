"""Compatibility wrapper for evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from jepa.config import load_and_resolve_config
from jepa.pipeline import evaluate_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper. Use evaluate.py.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scores", default=None)
    args = parser.parse_args()

    config = load_and_resolve_config(args.config)
    evaluate_stage(config, Path(args.run_dir), scores_path=args.scores)


if __name__ == "__main__":
    main()
