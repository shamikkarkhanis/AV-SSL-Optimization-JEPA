"""Aggregate results.jsonl from run_main_experiments.py into mean+/-std tables.

Produces:
  - a markdown table of the four conditions (mean +/- std over seeds)
  - a markdown table of the A/B/C baselines vs the best condition
  - <batch-root>/aggregate.json with the raw aggregates

Usage:
    python scripts/analyze_main_experiments.py --results experiments/main/run1/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

METRIC_KEYS = [
    "average_precision",
    "pr_auc",
    "ndcg",
    "precision_at_10",
    "recall_at_10",
    "best_val_loss",
    "total_train_time_seconds",
]


def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def _fmt(stat: Dict[str, Optional[float]], digits: int = 4) -> str:
    if stat["mean"] is None:
        return "-"
    if stat["n"] <= 1:
        return f"{stat['mean']:.{digits}f}"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def load_rows(results_path: Path) -> List[Dict[str, Any]]:
    # Accept a single results.jsonl OR a batch directory containing per-index
    # results_*.jsonl files (from a SLURM array run).
    if results_path.is_dir():
        files = sorted(results_path.glob("results*.jsonl"))
    else:
        files = [results_path]
    rows: List[Dict[str, Any]] = []
    for path in files:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    baseline_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("kind") == "train":
            train_groups[row["condition"]].append(row)
        elif row.get("kind") == "baseline":
            baseline_groups[row["method"]].append(row)

    def agg_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {"n_runs": len(group)}
        for key in METRIC_KEYS:
            out[key] = _mean_std([g["metrics"].get(key) for g in group])
        return out

    return {
        "conditions": {name: agg_group(g) for name, g in train_groups.items()},
        "baselines": {name: agg_group(g) for name, g in baseline_groups.items()},
    }


def render_markdown(agg: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("### Conditions (mean ± std over seeds)\n")
    lines.append("| Condition | n | AP | PR-AUC | NDCG | P@10 | R@10 | best val loss |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in ("samples_16", "samples_32", "sweeps_16", "sweeps_32"):
        if name not in agg["conditions"]:
            continue
        c = agg["conditions"][name]
        lines.append(
            f"| {name} | {c['n_runs']} | {_fmt(c['average_precision'])} | {_fmt(c['pr_auc'])} | "
            f"{_fmt(c['ndcg'])} | {_fmt(c['precision_at_10'])} | {_fmt(c['recall_at_10'])} | "
            f"{_fmt(c['best_val_loss'])} |"
        )

    lines.append("\n### Baselines vs trained method\n")
    lines.append("| Method | Trains a head? | n | AP | NDCG | P@10 |")
    lines.append("|---|---|---|---|---|---|")
    labels = {
        "embedding_density": ("B: embedding-density (kNN)", "no"),
        "masked_gap": ("A: masked-vs-clean gap", "no"),
        "untrained_predictor": ("C: untrained predictor", "no"),
    }
    for method in ("embedding_density", "masked_gap", "untrained_predictor"):
        if method not in agg["baselines"]:
            continue
        b = agg["baselines"][method]
        label, trains = labels[method]
        lines.append(
            f"| {label} | {trains} | {b['n_runs']} | {_fmt(b['average_precision'])} | "
            f"{_fmt(b['ndcg'])} | {_fmt(b['precision_at_10'])} |"
        )
    # Best trained condition for direct comparison.
    if agg["conditions"]:
        best_name = max(
            agg["conditions"],
            key=lambda n: agg["conditions"][n]["average_precision"]["mean"] or -1,
        )
        b = agg["conditions"][best_name]
        lines.append(
            f"| **Trained JEPA ({best_name})** | **yes** | {b['n_runs']} | "
            f"**{_fmt(b['average_precision'])}** | **{_fmt(b['ndcg'])}** | "
            f"**{_fmt(b['precision_at_10'])}** |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate main-experiment results.")
    parser.add_argument("--results", required=True, help="Path to results.jsonl")
    args = parser.parse_args()

    results_path = Path(args.results)
    rows = load_rows(results_path)
    agg = aggregate(rows)

    out_path = results_path.parent / "aggregate.json"
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(render_markdown(agg))
    print(f"\nWrote aggregates to {out_path}")


if __name__ == "__main__":
    main()
