"""Analyze factorial experiment results (main effects and 2-way interactions)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_rows(results_jsonl: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(results_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _parse_factorial_config(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_csv(path: Path, fieldnames: List[str], records: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _compute_main_effects(
    rows: List[Dict[str, Any]],
    factors: List[str],
    metric_key: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        metric_val = row.get("metrics", {}).get(metric_key)
        if metric_val is None:
            continue
        for factor in factors:
            level = row.get("factor_levels", {}).get(factor)
            grouped[(factor, str(level))].append(float(metric_val))

    out = []
    for (factor, level), values in sorted(grouped.items()):
        out.append(
            {
                "factor": factor,
                "level": level,
                "metric": metric_key,
                "n": len(values),
                "mean": _mean(values),
            }
        )
    return out


def _compute_two_way_interactions(
    rows: List[Dict[str, Any]],
    factors: List[str],
    metric_key: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, factor_a in enumerate(factors):
        for factor_b in factors[i + 1 :]:
            grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for row in rows:
                metric_val = row.get("metrics", {}).get(metric_key)
                if metric_val is None:
                    continue
                levels = row.get("factor_levels", {})
                a = str(levels.get(factor_a))
                b = str(levels.get(factor_b))
                grouped[(a, b)].append(float(metric_val))

            for (a, b), values in sorted(grouped.items()):
                out.append(
                    {
                        "factor_a": factor_a,
                        "level_a": a,
                        "factor_b": factor_b,
                        "level_b": b,
                        "metric": metric_key,
                        "n": len(values),
                        "mean": _mean(values),
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze JEPA factorial results")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to factorial results.jsonl",
    )
    parser.add_argument(
        "--factorial-config",
        default=None,
        help="Optional factorial config (to infer factor names and metrics)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=(
            "Metric keys to analyze. Defaults to response_metrics in factorial config "
            "or a built-in list if config is omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: sibling analysis/ folder next to results)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    rows = _load_rows(results_path)
    if not rows:
        raise ValueError("No rows found in results file.")

    cfg = {}
    if args.factorial_config:
        cfg = _parse_factorial_config(Path(args.factorial_config))

    if cfg.get("factors"):
        factors = list(cfg["factors"].keys())
    else:
        factors = sorted(rows[0].get("factor_levels", {}).keys())

    if args.metrics:
        metric_keys = args.metrics
    elif cfg.get("response_metrics"):
        metric_keys = list(cfg["response_metrics"])
    else:
        metric_keys = [
            "evaluation.mean_cosine_similarity",
            "inference.runtime_ms.mean",
            "inference.energy_joules.total",
        ]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_path.parent / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "results_jsonl": str(results_path),
        "num_rows": len(rows),
        "factors": factors,
        "metrics": metric_keys,
        "outputs": {},
    }

    for metric_key in metric_keys:
        slug = metric_key.replace(".", "_")

        main_effects = _compute_main_effects(rows, factors=factors, metric_key=metric_key)
        main_path = output_dir / f"main_effects__{slug}.csv"
        _write_csv(
            main_path,
            fieldnames=["factor", "level", "metric", "n", "mean"],
            records=main_effects,
        )

        interactions = _compute_two_way_interactions(
            rows, factors=factors, metric_key=metric_key
        )
        inter_path = output_dir / f"interactions_2way__{slug}.csv"
        _write_csv(
            inter_path,
            fieldnames=[
                "factor_a",
                "level_a",
                "factor_b",
                "level_b",
                "metric",
                "n",
                "mean",
            ],
            records=interactions,
        )

        summary["outputs"][metric_key] = {
            "main_effects_csv": str(main_path),
            "interactions_2way_csv": str(inter_path),
        }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Analysis complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()

