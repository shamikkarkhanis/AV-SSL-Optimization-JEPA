"""Tests for v1 config resolution and runtime profiles."""

from pathlib import Path

from jepa.config import load_and_resolve_config, resolve_config


def test_runtime_profile_cpu_applies_overrides():
    config = {
        "dataset": {},
        "model": {},
        "train": {},
        "score": {},
        "evaluation": {},
        "runtime": {
            "profile": "cpu",
            "batch_size_overrides": {"train": 8, "score": 2, "evaluation": 4},
            "profiles": {
                "cpu": {
                    "device": "cpu",
                    "amp": "none",
                    "num_workers": 0,
                    "batch_size_overrides": {"train": 1, "score": 1, "evaluation": 1},
                }
            },
        },
        "experiment": {},
    }
    resolved = resolve_config(config)
    assert resolved["runtime"]["device"] == "cpu"
    assert resolved["runtime"]["amp"] == "none"
    assert resolved["runtime"]["batch_size_overrides"]["train"] == 1


def test_load_and_resolve_legacy_config_maps_sections(tmp_path: Path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(
        """
seed: 7
data:
  manifest_path: data/manifests/clips_manifest.jsonl
  data_root: data/raw
  batch_size: 3
  num_workers: 2
training:
  epochs: 2
  device: cpu
model:
  encoder_name: foo/bar
  freeze_encoder: true
evaluation:
  batch_size: 5
inference:
  batch_size: 6
""",
        encoding="utf-8",
    )

    resolved = load_and_resolve_config(config_path)
    assert resolved["dataset"]["training_manifest"] == "data/manifests/clips_manifest.jsonl"
    assert resolved["runtime"]["batch_size_overrides"]["train"] == 3
    assert resolved["runtime"]["batch_size_overrides"]["score"] == 6
    assert resolved["runtime"]["batch_size_overrides"]["evaluation"] == 5
    assert resolved["model"]["encoder_mode"] == "frozen"
