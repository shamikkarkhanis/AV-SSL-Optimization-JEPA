from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from jepa.config import deep_merge, load_and_resolve_config
from jepa.pipeline import score_stage, write_root_summary


logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score clips from a checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--set", action="append", default=[], help="Inline JSON overrides for top-level config keys.")
    args = parser.parse_args()

    config = load_and_resolve_config(args.config)
    for override_json in args.set:
        config = deep_merge(config, json.loads(override_json))
    run_dir = Path(args.run_dir)
    scoring = score_stage(config, run_dir)
    write_root_summary(run_dir, None, scoring, None)


if __name__ == "__main__":
    main()
