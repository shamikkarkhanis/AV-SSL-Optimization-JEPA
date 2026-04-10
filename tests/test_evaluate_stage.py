"""Tests for evaluation-stage score/label separation."""

import json
from pathlib import Path

import pytest

from jepa.pipeline import evaluate_stage


def test_evaluate_stage_rejects_same_scores_and_labels_path(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "scoring").mkdir(parents=True)
    shared = run_dir / "scoring" / "scores.jsonl"
    shared.write_text(
        '{"clip_id":"clip-1","review_value_score":0.1,"mean_cosine_similarity":0.2}\n',
        encoding="utf-8",
    )

    config = {
        "seed": 42,
        "dataset": {
            "evaluation_labels": str(shared),
        },
        "model": {},
        "train": {},
        "score": {},
        "evaluation": {
            "primary_label_field": "binary_label",
        },
        "runtime": {
            "profile": "cpu",
            "profiles": {"cpu": {"device": "cpu", "amp": "none", "num_workers": 0}},
            "batch_size_overrides": {"train": 1, "score": 1, "evaluation": 1},
        },
        "experiment": {},
    }

    with pytest.raises(ValueError, match="different files"):
        evaluate_stage(config, run_dir)


def test_evaluate_stage_filters_scores_to_configured_evaluation_split(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "scoring").mkdir(parents=True)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"clip_id": "val-1", "frame_paths": ["a"], "split": "val"}),
                json.dumps({"clip_id": "test-1", "frame_paths": ["b"], "split": "test"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "scoring" / "scores.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"clip_id": "val-1", "split": "val", "review_value_score": 0.9, "mean_cosine_similarity": 0.1}),
                json.dumps({"clip_id": "test-1", "split": "test", "review_value_score": 0.2, "mean_cosine_similarity": 0.3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            [
                json.dumps({"clip_id": "val-1", "binary_label": 0}),
                json.dumps({"clip_id": "test-1", "binary_label": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "seed": 42,
        "dataset": {
            "evaluation_manifest": str(manifest),
            "evaluation_split": "test",
            "evaluation_labels": str(labels),
        },
        "model": {},
        "train": {},
        "score": {},
        "evaluation": {"primary_label_field": "binary_label", "k_values": [1]},
        "runtime": {
            "profile": "cpu",
            "profiles": {"cpu": {"device": "cpu", "amp": "none", "num_workers": 0}},
            "batch_size_overrides": {"train": 1, "score": 1, "evaluation": 1},
        },
        "experiment": {},
    }

    result = evaluate_stage(config, run_dir)
    assert result["ranking_metrics"]["num_joined_clips"] == 1
    assert result["ranking_metrics"]["positive_rate"] == 1.0
