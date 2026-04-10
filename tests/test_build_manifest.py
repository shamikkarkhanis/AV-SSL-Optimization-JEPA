"""Tests for building v1 manifests from nuScenes-style frame layouts."""

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_manifest", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_manifest = MODULE.build_manifest


def test_build_manifest_writes_v1_schema(tmp_path: Path):
    cam_dir = tmp_path / "v1.0-mini" / "samples" / "CAM_FRONT"
    cam_dir.mkdir(parents=True)
    for idx in range(8):
        timestamp = 1000 + idx
        frame_name = f"scene-a__CAM_FRONT__{timestamp}.jpg"
        (cam_dir / frame_name).write_bytes(b"frame")

    output = tmp_path / "manifest.jsonl"
    count = build_manifest(
        dataroot=tmp_path / "v1.0-mini",
        output_path=output,
        window_size=4,
        stride=4,
    )

    assert count == 2
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["clip_id"] == "scene-a__CAM_FRONT__1000__1003"
    assert rows[0]["split"] in {"train", "val", "test"}
    assert rows[0]["scene_id"] == "scene-a"
    assert rows[0]["camera"] == "CAM_FRONT"
    assert rows[0]["frame_paths"][0] == "samples/CAM_FRONT/scene-a__CAM_FRONT__1000.jpg"
    assert rows[0]["timestamps"] == [1000, 1001, 1002, 1003]
    assert rows[0]["metadata"]["source_dataset"] == "v1.0-mini"


def test_build_manifest_writes_eval_labels_for_val_and_test(tmp_path: Path):
    cam_dir = tmp_path / "v1.0-mini" / "samples" / "CAM_FRONT"
    cam_dir.mkdir(parents=True)
    for scene_name in ("scene-a", "scene-b", "scene-c"):
        for idx in range(4):
            timestamp = 1000 + idx
            frame_name = f"{scene_name}__CAM_FRONT__{timestamp}.jpg"
            (cam_dir / frame_name).write_bytes(b"frame")

    output = tmp_path / "manifest.jsonl"
    build_manifest(
        dataroot=tmp_path / "v1.0-mini",
        output_path=output,
        window_size=4,
        stride=4,
        train_ratio=1 / 3,
        val_ratio=1 / 3,
        seed=1,
    )

    labels_path = tmp_path / "manifest_evaluation_labels.jsonl"
    rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["split"] for row in rows} == {"val", "test"}
    assert all("binary_label" in row for row in rows)
