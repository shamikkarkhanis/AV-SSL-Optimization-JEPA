"""Tests for factorial design utilities."""

from jepa.experiments.design import (
    apply_overrides,
    build_full_factorial_runs,
    set_dotted_path,
)


def test_set_dotted_path_creates_nested_keys():
    cfg = {}
    set_dotted_path(cfg, "a.b.c", 7)
    assert cfg == {"a": {"b": {"c": 7}}}


def test_apply_overrides_sets_levels_and_seed():
    base = {"data": {"mask_ratio": 0.75}, "seed": 42}
    out = apply_overrides(
        base_config=base,
        factor_levels={"data.mask_ratio": 0.5, "training.lr": 1e-4},
        replicate_seed=123,
    )
    assert out["data"]["mask_ratio"] == 0.5
    assert out["training"]["lr"] == 1e-4
    assert out["seed"] == 123
    # Ensure base is not mutated
    assert base["data"]["mask_ratio"] == 0.75


def test_build_full_factorial_runs_size_and_ids():
    runs = build_full_factorial_runs(
        factors={"a.x": [1, 2], "b.y": ["u", "v"]},
        replicates=[11, 22],
    )
    # 2 x 2 x 2 replicates
    assert len(runs) == 8
    assert runs[0]["run_id"] == 1
    assert runs[-1]["run_id"] == 8
    assert runs[0]["combination_id"] == 1
    assert runs[-1]["combination_id"] == 4
