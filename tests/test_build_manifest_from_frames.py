"""Tests for building manifests from extracted frames."""

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_manifest_from_frames.py"
SPEC = importlib.util.spec_from_file_location("build_manifest_from_frames", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_manifest_from_frames = MODULE.build_manifest_from_frames


def test_build_manifest_from_scene_camera_layout(tmp_path: Path):
    root = tmp_path / "frames"
    for scene in ("scene_a", "scene_b"):
        camera_dir = root / scene / "CAM_FRONT"
        camera_dir.mkdir(parents=True)
        for idx in range(8):
            (camera_dir / f"{idx:06d}.jpg").write_bytes(b"frame")

    output = tmp_path / "manifest.jsonl"
    count = build_manifest_from_frames(
        frames_root=root,
        output_path=output,
        clip_length=4,
        stride=4,
        train_ratio=0.5,
        val_ratio=0.5,
        seed=1,
    )

    assert count == 4
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["clip_id"].startswith("scene_")
    assert rows[0]["camera"] == "CAM_FRONT"
    assert len(rows[0]["frame_paths"]) == 4
    assert rows[0]["frame_paths"][0].startswith("scene_")
    assert rows[0]["split"] in {"train", "val", "test"}
