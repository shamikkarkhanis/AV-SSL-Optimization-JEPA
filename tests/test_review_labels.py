"""Tests for the local label-review helper."""

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "review_labels.py"
SPEC = importlib.util.spec_from_file_location("review_labels", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ReviewStore = MODULE.ReviewStore
infer_manifest_path = MODULE.infer_manifest_path
_build_clip_gif_bytes = MODULE._build_clip_gif_bytes
_resolve_frame_path = MODULE._resolve_frame_path
update_label_file = MODULE.update_label_file


def test_infer_manifest_path_prefers_jsonl_then_py(tmp_path: Path):
    labels = tmp_path / "baseline_manifest_evaluation_labels.jsonl"
    labels.write_text("", encoding="utf-8")
    manifest = tmp_path / "baseline_manifest.py"
    manifest.write_text("", encoding="utf-8")

    assert infer_manifest_path(labels) == manifest


def test_update_label_file_persists_binary_label(tmp_path: Path):
    labels_path = tmp_path / "labels.jsonl"
    rows = [{"clip_id": "clip-1", "binary_label": None}]
    labels_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    update_label_file(labels_path, rows, "clip-1", 1)

    persisted = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    assert persisted[0]["binary_label"] == 1


def test_review_store_joins_manifest_and_labels_and_updates_counts(tmp_path: Path):
    labels_path = tmp_path / "baseline_manifest_evaluation_labels.jsonl"
    labels_path.write_text(
        "\n".join(
            [
                json.dumps({"clip_id": "clip-1", "binary_label": None, "split": "test"}),
                json.dumps({"clip_id": "clip-2", "binary_label": 0, "split": "val"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "baseline_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({"clip_id": "clip-1", "split": "test", "scene_id": "scene-a", "camera": "CAM_FRONT", "frame_paths": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]}),
                json.dumps({"clip_id": "clip-2", "split": "val", "scene_id": "scene-b", "camera": "CAM_FRONT", "frame_paths": ["f.jpg", "g.jpg"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store = ReviewStore(labels_path=labels_path, manifest_path=manifest_path)
    assert store.first_unlabeled_index() == 0

    payload = store.clip_payload(0)
    assert payload["counts"]["unlabeled"] == 1
    assert len(payload["frames"]) == 4

    store.set_label(0, 1)
    updated_payload = store.clip_payload(0)
    assert updated_payload["binary_label"] == 1
    assert updated_payload["counts"]["positives"] == 1


def test_resolve_frame_path_tries_samples_subdirectory(tmp_path: Path):
    data_root = tmp_path / "v1.0-mini"
    frame = data_root / "samples" / "CAM_FRONT" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_text("x", encoding="utf-8")

    resolved = _resolve_frame_path("CAM_FRONT/frame.jpg", data_root, tmp_path / "manifest.jsonl")
    assert resolved == frame


def test_build_clip_gif_bytes_from_manifest_frames(tmp_path: Path):
    data_root = tmp_path / "dataset"
    frame_dir = data_root / "samples" / "CAM_FRONT"
    frame_dir.mkdir(parents=True)
    frame_paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        frame_path = frame_dir / f"frame_{index}.jpg"
        MODULE.Image.new("RGB", (64, 32), color=color).save(frame_path)
        frame_paths.append(f"CAM_FRONT/{frame_path.name}")

    gif_bytes = _build_clip_gif_bytes(frame_paths, data_root, tmp_path / "manifest.jsonl")

    assert gif_bytes[:6] in (b"GIF87a", b"GIF89a")


def test_review_store_clip_payload_includes_clip_url_and_cacheable_gif(tmp_path: Path):
    labels_path = tmp_path / "baseline_manifest_evaluation_labels.jsonl"
    labels_path.write_text(
        json.dumps({"clip_id": "clip-1", "binary_label": None, "split": "test"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "baseline_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "clip_id": "clip-1",
                "split": "test",
                "scene_id": "scene-a",
                "camera": "CAM_FRONT",
                "frame_paths": ["CAM_FRONT/frame.jpg"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frame_dir = tmp_path / "dataset" / "samples" / "CAM_FRONT"
    frame_dir.mkdir(parents=True)
    MODULE.Image.new("RGB", (32, 32), color=(255, 0, 0)).save(frame_dir / "frame.jpg")

    store = ReviewStore(labels_path=labels_path, manifest_path=manifest_path, data_root=tmp_path / "dataset")
    payload = store.clip_payload(0)

    assert payload["clip_url"] == "/clip?clip_id=clip-1"
    assert store.clip_gif_bytes("clip-1")[:6] in (b"GIF87a", b"GIF89a")
