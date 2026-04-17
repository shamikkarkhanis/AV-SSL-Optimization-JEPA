"""Tests for scene-level manifest re-splitting."""

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "resplit_manifest.py"
SPEC = importlib.util.spec_from_file_location("resplit_manifest", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
resplit_manifest = MODULE.resplit_manifest


def test_resplit_manifest_moves_test_scenes_to_val_and_updates_labels(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    output_manifest = tmp_path / "manifest_resplit.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output_labels = tmp_path / "labels_resplit.jsonl"

    manifest_rows = [
        {"clip_id": "train-1", "split": "train", "scene_id": "scene-train", "frame_paths": ["a.jpg"]},
        {"clip_id": "test-a-1", "split": "test", "scene_id": "scene-a", "frame_paths": ["b.jpg"]},
        {"clip_id": "test-a-2", "split": "test", "scene_id": "scene-a", "frame_paths": ["c.jpg"]},
        {"clip_id": "test-b-1", "split": "test", "scene_id": "scene-b", "frame_paths": ["d.jpg"]},
        {"clip_id": "test-b-2", "split": "test", "scene_id": "scene-b", "frame_paths": ["e.jpg"]},
    ]
    label_rows = [
        {"clip_id": "test-a-1", "split": "test", "scene_id": "scene-a", "binary_label": None},
        {"clip_id": "test-a-2", "split": "test", "scene_id": "scene-a", "binary_label": 1},
        {"clip_id": "test-b-1", "split": "test", "scene_id": "scene-b", "binary_label": 0},
        {"clip_id": "test-b-2", "split": "test", "scene_id": "scene-b", "binary_label": None},
    ]

    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    labels_path.write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows),
        encoding="utf-8",
    )

    result = resplit_manifest(
        input_path=manifest_path,
        output_path=output_manifest,
        val_fraction_of_test_scenes=0.5,
        seed=1,
        labels_input_path=labels_path,
        labels_output_path=output_labels,
    )

    rewritten_manifest = [
        json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines()
    ]
    rewritten_labels = [
        json.loads(line) for line in output_labels.read_text(encoding="utf-8").splitlines()
    ]

    val_scenes = {row["scene_id"] for row in rewritten_manifest if row["split"] == "val"}
    assert len(val_scenes) == 1
    assert result["reassigned_scenes"] == 1
    assert result["reassigned_clips"] == 2
    assert result["reassigned_label_rows"] == 2
    assert {row["scene_id"] for row in rewritten_labels if row["split"] == "val"} == val_scenes
    assert rewritten_manifest[0]["split"] == "train"


def test_resplit_manifest_requires_fraction_in_range(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    output_manifest = tmp_path / "manifest_resplit.jsonl"
    manifest_path.write_text(
        json.dumps({"clip_id": "test-1", "split": "test", "scene_id": "scene-a", "frame_paths": ["a.jpg"]}) + "\n",
        encoding="utf-8",
    )

    try:
        resplit_manifest(
            input_path=manifest_path,
            output_path=output_manifest,
            val_fraction_of_test_scenes=1.5,
        )
    except ValueError as exc:
        assert "between 0.0 and 1.0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-range fraction.")
