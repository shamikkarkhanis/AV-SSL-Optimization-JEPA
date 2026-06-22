"""Tests for per-stage frames_per_clip (held-fixed evaluation benchmark)."""

from jepa.pipeline import _stage_frames_per_clip


def test_all_stages_default_to_global_frames_per_clip():
    cfg = {"frames_per_clip": 32}
    for stage in ("train", "validation", "score", "evaluation"):
        assert _stage_frames_per_clip(cfg, stage) == 32


def test_scoring_and_evaluation_can_override_while_training_varies():
    cfg = {
        "frames_per_clip": 32,
        "scoring_frames_per_clip": 16,
        "evaluation_frames_per_clip": 16,
    }
    assert _stage_frames_per_clip(cfg, "train") == 32
    assert _stage_frames_per_clip(cfg, "validation") == 32
    assert _stage_frames_per_clip(cfg, "score") == 16
    assert _stage_frames_per_clip(cfg, "evaluation") == 16


def test_missing_global_defaults_to_16():
    assert _stage_frames_per_clip({}, "train") == 16
