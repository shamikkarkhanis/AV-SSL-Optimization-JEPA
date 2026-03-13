"""Tests for legacy manifest migration."""

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "migrate_manifest.py"
SPEC = importlib.util.spec_from_file_location("migrate_manifest", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
migrate_manifest = MODULE.migrate_manifest


def test_migrate_manifest_adds_clip_id_and_split(tmp_path: Path):
    input_path = tmp_path / "legacy.jsonl"
    output_path = tmp_path / "v1.jsonl"
    records = [
        {
            "scene": "scene-a",
            "camera": "CAM_FRONT",
            "frames": [f"a_{idx}.jpg" for idx in range(16)],
            "timestamps": list(range(16)),
        },
        {
            "scene": "scene-b",
            "camera": "CAM_FRONT",
            "frames": [f"b_{idx}.jpg" for idx in range(16)],
            "timestamps": list(range(16, 32)),
        },
    ]
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    migrated_count = migrate_manifest(input_path, output_path, train_ratio=0.5, score_ratio=0.5)
    assert migrated_count == 2

    with output_path.open("r", encoding="utf-8") as handle:
        migrated = [json.loads(line) for line in handle if line.strip()]

    assert migrated[0]["clip_id"].startswith("scene-a__CAM_FRONT__")
    assert migrated[0]["frame_paths"][0] == "a_0.jpg"
    assert migrated[0]["split"] in {"train", "val", "test"}
    assert migrated[0]["scene_id"] == "scene-a"
