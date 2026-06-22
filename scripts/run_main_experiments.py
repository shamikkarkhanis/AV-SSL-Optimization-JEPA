"""Driver for the FTC 2026 experiment cycle.

Runs the source x frames x seeds matrix on a FIXED labeled benchmark, plus the
three no-training baselines (A: masked_gap, B: embedding_density, C:
untrained_predictor). Every item in the plan is indexed so it maps directly onto
a SLURM job array (one GPU task per index) or can be run sequentially.

Usage
-----
    # Inspect the full plan (no compute):
    python scripts/run_main_experiments.py --dry-run

    # Run a single plan item (for `sbatch --array`):
    python scripts/run_main_experiments.py --index $SLURM_ARRAY_TASK_ID \
        --batch-root experiments/main/$SLURM_JOB_ID

    # Run everything sequentially on one GPU:
    python scripts/run_main_experiments.py --batch-root experiments/main/run1

The four training conditions reproduce the report's A-D matrix but with longer
training and multiple seeds. They differ ONLY in training manifest +
frames_per_clip; scoring/evaluation use the fixed benchmark from the base config.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jepa.baselines import score_baseline_stage
from jepa.config import deep_merge, load_and_resolve_config
from jepa.pipeline import evaluate_stage, run_experiment, write_root_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_main_experiments")


# Four training conditions = the report's A-D matrix.
# train_manifest is resolved against --manifest-dir (or used as-is if absolute).
CONDITIONS: List[Dict[str, Any]] = [
    {"name": "samples_16", "source": "samples", "frames": 16, "train_manifest": "nuscenes_samples_16.jsonl"},
    {"name": "samples_32", "source": "samples", "frames": 32, "train_manifest": "nuscenes_samples_32.jsonl"},
    {"name": "sweeps_16",  "source": "sweeps",  "frames": 16, "train_manifest": "nuscenes_sweeps_16.jsonl"},
    {"name": "sweeps_32",  "source": "sweeps",  "frames": 32, "train_manifest": "nuscenes_sweeps_32.jsonl"},
    # Data-scaling probe: trainval01-04 sweeps (~5x sweeps_16). Run separately with
    # fewer epochs (it is ~40 min/epoch) to test whether more data helps.
    {"name": "sweeps_all_16", "source": "sweeps_all", "frames": 16, "train_manifest": "nuscenes_sweeps_all_16.jsonl"},
]

# Deterministic baselines run once; the random-init baseline (C) runs per seed.
DETERMINISTIC_BASELINES = ("masked_gap", "embedding_density")
SEEDED_BASELINES = ("untrained_predictor",)


def build_plan(
    conditions: List[Dict[str, Any]],
    seeds: List[int],
    include_baselines: bool,
    baseline_seeds: List[int],
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for cond in conditions:
        for seed in seeds:
            plan.append({"kind": "train", "condition": cond["name"], "seed": seed})
    if include_baselines:
        for method in DETERMINISTIC_BASELINES:
            plan.append({"kind": "baseline", "method": method, "seed": seeds[0]})
        for method in SEEDED_BASELINES:
            for seed in baseline_seeds:
                plan.append({"kind": "baseline", "method": method, "seed": seed})
    return plan


def _condition(name: str) -> Dict[str, Any]:
    for cond in CONDITIONS:
        if cond["name"] == name:
            return cond
    raise KeyError(f"Unknown condition {name!r}")


def _resolve_manifest(manifest: str, manifest_dir: Optional[str]) -> str:
    path = Path(manifest)
    if path.is_absolute() or manifest_dir is None:
        return str(path)
    return str(Path(manifest_dir) / manifest)


def build_train_config(
    base_config: Dict[str, Any],
    cond: Dict[str, Any],
    seed: int,
    epochs: Optional[int],
    manifest_dir: Optional[str],
    data_root: Optional[str],
) -> Dict[str, Any]:
    train_manifest = _resolve_manifest(cond["train_manifest"], manifest_dir)
    overrides: Dict[str, Any] = {
        "seed": seed,
        "dataset": {
            "training_manifest": train_manifest,
            "validation_manifest": train_manifest,
            "frames_per_clip": cond["frames"],
        },
        "experiment": {"name": f"{cond['name']}__seed{seed}"},
    }
    if data_root is not None:
        overrides["dataset"]["data_root"] = data_root
    if epochs is not None:
        overrides["train"] = {"epochs": int(epochs)}
    return deep_merge(deepcopy(base_config), overrides)


def build_baseline_config(base_config: Dict[str, Any], seed: int, name: str) -> Dict[str, Any]:
    return deep_merge(
        deepcopy(base_config),
        {"seed": seed, "experiment": {"name": name}},
    )


def extract_metrics(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = run_summary.get("evaluation") or {}
    ranking = evaluation.get("ranking_metrics", {}) or {}
    scoring = run_summary.get("scoring") or {}
    training = run_summary.get("training") or {}
    return {
        "average_precision": ranking.get("average_precision"),
        "pr_auc": ranking.get("pr_auc"),
        "ndcg": ranking.get("ndcg"),
        "precision_at_10": (ranking.get("precision_at_k", {}) or {}).get("10"),
        "recall_at_10": (ranking.get("recall_at_k", {}) or {}).get("10"),
        "positive_rate": ranking.get("positive_rate"),
        "num_joined_clips": ranking.get("num_joined_clips"),
        "best_val_loss": training.get("best_val_loss"),
        "final_val_loss": training.get("final_val_loss"),
        "total_train_time_seconds": training.get("total_train_time_seconds"),
        "clips_per_second": scoring.get("clips_per_second"),
        "latency_ms_mean": (scoring.get("latency_ms", {}) or {}).get("mean"),
    }


def run_item(
    item: Dict[str, Any],
    base_config: Dict[str, Any],
    batch_root: Path,
    epochs: Optional[int],
    manifest_dir: Optional[str],
    data_root: Optional[str],
    baseline_k: int,
) -> Dict[str, Any]:
    if item["kind"] == "train":
        cond = _condition(item["condition"])
        run_name = f"{cond['name']}__seed{item['seed']}"
        run_dir = batch_root / "runs" / run_name
        config = build_train_config(base_config, cond, item["seed"], epochs, manifest_dir, data_root)
        summary = run_experiment(config, run_dir)
        return {
            "kind": "train",
            "condition": cond["name"],
            "source": cond["source"],
            "frames": cond["frames"],
            "seed": item["seed"],
            "run_dir": str(run_dir),
            "metrics": extract_metrics(summary),
        }

    # baseline
    method = item["method"]
    run_name = f"baseline_{method}__seed{item['seed']}"
    run_dir = batch_root / "runs" / run_name
    config = build_baseline_config(base_config, item["seed"], run_name)
    scoring = score_baseline_stage(config, run_dir, method=method, k=baseline_k)
    evaluation = evaluate_stage(config, run_dir)
    summary = write_root_summary(run_dir, None, scoring, evaluation)
    return {
        "kind": "baseline",
        "method": method,
        "seed": item["seed"],
        "run_dir": str(run_dir),
        "metrics": extract_metrics(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FTC 2026 experiment driver.")
    parser.add_argument("--base-config", default="configs/benchmark.yaml")
    parser.add_argument("--batch-root", default="experiments/main/batch")
    parser.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated training seeds.")
    parser.add_argument(
        "--baseline-seeds",
        default="1,2,3",
        help="Seeds for the random-init untrained-predictor baseline.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--baseline-k", type=int, default=5, help="kNN neighbours for embedding_density.")
    parser.add_argument("--conditions", default=None, help="Comma-separated subset of condition names.")
    parser.add_argument("--no-baselines", action="store_true", help="Skip the A/B/C baselines.")
    parser.add_argument("--manifest-dir", default=None, help="Directory holding the training manifests.")
    parser.add_argument("--data-root", default=None, help="Override dataset.data_root for training conditions.")
    parser.add_argument("--index", type=int, default=None, help="Run only this plan index (for SLURM arrays).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    baseline_seeds = [int(s) for s in args.baseline_seeds.split(",") if s.strip()]
    conditions = CONDITIONS
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in CONDITIONS if c["name"] in wanted]

    plan = build_plan(conditions, seeds, not args.no_baselines, baseline_seeds)

    if args.dry_run:
        for i, item in enumerate(plan):
            print(f"[{i}] {item}")
        print(f"\nTotal plan items: {len(plan)}")
        return

    base_config = load_and_resolve_config(args.base_config)
    batch_root = Path(args.batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    # When running a single index (SLURM array), write a per-index results file so
    # concurrent array tasks never append to the same file on a shared filesystem.
    # analyze_main_experiments.py merges results*.jsonl back together.
    results_path = (
        batch_root / f"results_{args.index:03d}.jsonl"
        if args.index is not None
        else batch_root / "results.jsonl"
    )

    # Persist the plan for reproducibility / array bookkeeping.
    with open(batch_root / "plan.jsonl", "w") as f:
        for i, item in enumerate(plan):
            f.write(json.dumps({"index": i, **item}) + "\n")

    indices = [args.index] if args.index is not None else list(range(len(plan)))
    for i in indices:
        item = plan[i]
        logger.info("[%d/%d] running %s", i, len(plan), item)
        start = time.time()
        row = run_item(item, base_config, batch_root, args.epochs, args.manifest_dir, args.data_root, args.baseline_k)
        row["index"] = i
        row["elapsed_seconds"] = time.time() - start
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info("[%d] done in %.1fs: AP=%s", i, row["elapsed_seconds"], row["metrics"].get("average_precision"))


if __name__ == "__main__":
    main()
