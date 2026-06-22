"""Run a no-training baseline (A/B/C) and evaluate it against the benchmark.

Examples
--------
    python scripts/score_baselines.py --config configs/benchmark.yaml \
        --run-dir experiments/runs/baseline_masked_gap --method masked_gap

    python scripts/score_baselines.py --config configs/benchmark.yaml \
        --run-dir experiments/runs/baseline_density --method embedding_density --k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jepa.baselines import BASELINE_METHODS, score_baseline_stage
from jepa.config import deep_merge, load_and_resolve_config
from jepa.pipeline import evaluate_stage, write_root_summary

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score clips with a no-training baseline and evaluate the ranking."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=BASELINE_METHODS,
        help="masked_gap (A), embedding_density (B), untrained_predictor (C)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Neighbour count for embedding_density (ignored otherwise).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Only score; skip the evaluation stage.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Inline JSON overrides for top-level config keys.",
    )
    args = parser.parse_args()

    config = load_and_resolve_config(args.config)
    for override_json in args.set:
        config = deep_merge(config, json.loads(override_json))

    run_dir = Path(args.run_dir)
    scoring = score_baseline_stage(config, run_dir, method=args.method, k=args.k)
    evaluation = None
    if not args.no_eval:
        evaluation = evaluate_stage(config, run_dir)
    write_root_summary(run_dir, None, scoring, evaluation)

    if evaluation is not None:
        metrics = evaluation.get("ranking_metrics", {})
        logging.info(
            "[%s] AP=%.4f NDCG=%.4f P@10=%s",
            args.method,
            metrics.get("average_precision", float("nan")),
            metrics.get("ndcg", float("nan")),
            metrics.get("precision_at_k", {}).get("10"),
        )


if __name__ == "__main__":
    main()
