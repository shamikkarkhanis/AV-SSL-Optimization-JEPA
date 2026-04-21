"""Run full-factorial experiment batches using the stage-based pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from jepa.experiments import apply_overrides, build_full_factorial_runs
from jepa.config import load_and_resolve_config
from jepa.pipeline import run_experiment


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_factorial")


def _safe_value(value: Any) -> str:
    text = str(value)
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(out)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_hypothesis_name(run_spec: Dict[str, Any]) -> str:
    parts = [f"f{run_spec['combination_id']:03d}", f"r{run_spec['replicate_id']:02d}"]
    for factor, value in sorted(run_spec["factor_levels"].items()):
        leaf = factor.split(".")[-1]
        parts.append(f"{leaf}-{_safe_value(value)}")
    return _safe_name("__".join(parts))


def _extract_metrics(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    evaluation_summary = run_summary.get("evaluation", {}) or {}
    ranking_metrics = evaluation_summary.get("ranking_metrics", {}) or {}
    model_health = evaluation_summary.get("model_health", {}) or {}
    scoring_summary = run_summary.get("scoring", {}) or {}
    training_summary = run_summary.get("training", {}) or {}
    return {
        "evaluation.average_precision": ranking_metrics.get("average_precision"),
        "evaluation.precision_at_10": (ranking_metrics.get("precision_at_k", {}) or {}).get("10"),
        "evaluation.recall_at_10": (ranking_metrics.get("recall_at_k", {}) or {}).get("10"),
        "evaluation.ndcg": ranking_metrics.get("ndcg"),
        "evaluation.mean_cosine_similarity": model_health.get("mean_cosine_similarity"),
        "scoring.clips_per_second": scoring_summary.get("clips_per_second"),
        "scoring.latency_ms.mean": (scoring_summary.get("latency_ms", {}) or {}).get("mean"),
        "scoring.estimated_energy_joules": scoring_summary.get("estimated_energy_joules"),
        "training.total_train_time_seconds": training_summary.get("total_train_time_seconds"),
        "training.peak_memory_mb": training_summary.get("peak_memory_mb"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full factorial JEPA experiments")
    parser.add_argument(
        "--config",
        default="configs/factorial.yaml",
        help="Path to factorial design config YAML",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional batch run directory override. Individual runs will be created under <run-dir>/runs/.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on total runs (for quick debugging)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and print design matrix without executing training runs",
    )
    parser.add_argument(
        "--design-id",
        default=None,
        help="Optional design identifier for tracking (e.g., fx_v2)",
    )
    parser.add_argument(
        "--hypothesis",
        default=None,
        help="Explicit hypothesis statement for this factorial batch",
    )
    parser.add_argument(
        "--primary-metric",
        default=None,
        help="Primary metric key used for final decision",
    )
    parser.add_argument(
        "--decision-rule",
        default=None,
        help="Decision rule text (e.g., maximize cosine under runtime budget)",
    )
    args = parser.parse_args()

    factorial_cfg_path = Path(args.config)
    factorial_cfg = _load_yaml(factorial_cfg_path)

    base_config_path = Path(factorial_cfg.get("base_config", "configs/default.yaml"))
    base_config = load_and_resolve_config(base_config_path)
    experiments_root = str(
        factorial_cfg.get("experiments_root", "experiments/factorial_runs")
    )
    factors = factorial_cfg.get("factors", {})
    replicates = factorial_cfg.get("replicates", [int(base_config.get("seed", 42))])
    design_type = str(factorial_cfg.get("design_type", "full")).lower()
    if design_type != "full":
        raise ValueError("Only design_type='full' is currently supported.")

    run_specs = build_full_factorial_runs(factors=factors, replicates=replicates)
    if args.max_runs is not None:
        run_specs = run_specs[: max(args.max_runs, 0)]

    now = datetime.now()
    day_bucket = now.strftime("%Y-%m-%d")
    batch_id = now.strftime("batch_%H%M%S")
    batch_root = (
        Path(args.run_dir)
        if args.run_dir
        else Path(experiments_root) / day_bucket / batch_id
    )
    batch_root.mkdir(parents=True, exist_ok=True)
    design_matrix_path = batch_root / "design_matrix.jsonl"
    results_path = batch_root / "results.jsonl"
    batch_summary_path = batch_root / "batch_summary.json"

    cli_tracking = {
        "design_id": args.design_id,
        "hypothesis": args.hypothesis,
        "primary_metric": args.primary_metric,
        "decision_rule": args.decision_rule,
    }
    config_tracking = factorial_cfg.get("tracking", {}) if isinstance(factorial_cfg, dict) else {}
    tracking = {
        "design_id": cli_tracking["design_id"] or config_tracking.get("design_id"),
        "hypothesis": cli_tracking["hypothesis"] or config_tracking.get("hypothesis"),
        "primary_metric": cli_tracking["primary_metric"] or config_tracking.get("primary_metric"),
        "decision_rule": cli_tracking["decision_rule"] or config_tracking.get("decision_rule"),
    }

    with open(batch_root / "factorial_config_resolved.yaml", "w") as f:
        yaml.safe_dump(factorial_cfg, f, sort_keys=False)

    with open(design_matrix_path, "w") as f:
        for spec in run_specs:
            f.write(json.dumps(spec) + "\n")

    batch_summary = {
        "status": "planned",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batch_root": str(batch_root),
        "factorial_config_path": str(factorial_cfg_path),
        "factorial_config_sha256": _file_sha256(factorial_cfg_path),
        "base_config_path": str(base_config_path),
        "base_config_sha256": _file_sha256(base_config_path),
        "tracking": tracking,
        "design_type": design_type,
        "num_runs_planned": len(run_specs),
        "factors": factors,
        "replicates": replicates,
        "results_jsonl": str(results_path),
        "design_matrix_jsonl": str(design_matrix_path),
        "completed_runs": 0,
        "failed_runs": 0,
    }
    with open(batch_summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    logger.info("Prepared %d runs. Design matrix: %s", len(run_specs), design_matrix_path)
    if args.dry_run:
        batch_summary["status"] = "dry_run_complete"
        with open(batch_summary_path, "w") as f:
            json.dump(batch_summary, f, indent=2)
        return

    batch_summary["status"] = "running"
    with open(batch_summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    for idx, spec in enumerate(run_specs, start=1):
        hypothesis_name = _build_hypothesis_name(spec)
        run_config = apply_overrides(
            base_config=base_config,
            factor_levels=spec["factor_levels"],
            replicate_seed=spec["replicate_seed"],
        )
        run_config.setdefault("experiment", {})
        run_config["experiment"]["name"] = hypothesis_name
        run_config["experiment"]["output_root"] = experiments_root
        run_dir = batch_root / "runs" / hypothesis_name

        run_start = time.time()
        logger.info(
            "[%d/%d] Running %s with factors=%s seed=%s",
            idx,
            len(run_specs),
            hypothesis_name,
            spec["factor_levels"],
            spec["replicate_seed"],
        )
        run_summary = run_experiment(config=run_config, run_dir=run_dir)
        elapsed_s = time.time() - run_start

        row = {
            "run_id": spec["run_id"],
            "combination_id": spec["combination_id"],
            "replicate_id": spec["replicate_id"],
            "replicate_seed": spec["replicate_seed"],
            "tracking": tracking,
            "factor_levels": spec["factor_levels"],
            "hypothesis_name": hypothesis_name,
            "elapsed_seconds": elapsed_s,
            "run_dir": str(run_dir),
            "run_summary_json": str(run_dir / "summary.json"),
            "metrics": _extract_metrics(run_summary),
        }
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        batch_summary["completed_runs"] = idx
        with open(batch_summary_path, "w") as f:
            json.dump(batch_summary, f, indent=2)

    batch_summary["status"] = "completed"
    batch_summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    logger.info("Factorial batch complete. Results: %s", results_path)
    with open(batch_summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)


if __name__ == "__main__":
    main()
