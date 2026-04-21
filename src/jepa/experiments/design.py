"""Factorial design utilities."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any, Dict, Iterable, List


def set_dotted_path(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set nested dictionary value by dot-separated path."""
    keys = dotted_path.split(".")
    curr = target
    for key in keys[:-1]:
        next_value = curr.get(key)
        if not isinstance(next_value, dict):
            curr[key] = {}
        curr = curr[key]
    curr[keys[-1]] = value


def apply_overrides(
    base_config: Dict[str, Any],
    factor_levels: Dict[str, Any],
    replicate_seed: int | None = None,
) -> Dict[str, Any]:
    """Build a resolved run config by applying factor-level overrides."""
    resolved = deepcopy(base_config)
    for path, value in factor_levels.items():
        set_dotted_path(resolved, path, value)
    if replicate_seed is not None:
        resolved["seed"] = int(replicate_seed)
    return resolved


def _factor_product(factors: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    if not factors:
        return [{}]
    keys = list(factors.keys())
    values_list = [list(factors[key]) for key in keys]
    combos = []
    for values in product(*values_list):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def build_full_factorial_runs(
    factors: Dict[str, Iterable[Any]],
    replicates: List[int] | None = None,
) -> List[Dict[str, Any]]:
    """Create run specifications for a full-factorial design."""
    combos = _factor_product(factors)
    replicate_values = replicates if replicates else [None]
    runs: List[Dict[str, Any]] = []

    run_id = 1
    for combo_idx, combo in enumerate(combos, start=1):
        for rep_idx, rep_seed in enumerate(replicate_values, start=1):
            runs.append(
                {
                    "run_id": run_id,
                    "combination_id": combo_idx,
                    "replicate_id": rep_idx,
                    "replicate_seed": rep_seed,
                    "factor_levels": combo,
                }
            )
            run_id += 1
    return runs

