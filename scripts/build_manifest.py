"""Build a v1 JSONL clip manifest from a nuScenes-style data directory."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_manifest")


def parse_scene_and_ts(filename: str, camera: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract scene_id and timestamp from filename.

    Format expected: {scene_id}__{camera}__{timestamp}.jpg
    Example: n008-2018-08-01...__CAM_FRONT__1533151603512404.jpg
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    marker = f"__{camera}__"

    if marker not in name:
        return None, None

    try:
        left, right = name.split(marker, 1)
        ts = int(right)
        return left, ts
    except ValueError:
        return None, None


def _assign_scene_splits(
    scene_ids,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    unique_scenes = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_scenes)

    num_scenes = len(unique_scenes)
    train_cut = int(num_scenes * train_ratio)
    val_cut = train_cut + int(num_scenes * val_ratio)

    split_map: Dict[str, str] = {}
    for index, scene_id in enumerate(unique_scenes):
        if index < train_cut:
            split_map[scene_id] = "train"
        elif index < val_cut:
            split_map[scene_id] = "val"
        else:
            split_map[scene_id] = "test"
    return split_map


def _resolve_camera_dir(root: Path, camera: str) -> Tuple[Path, Path]:
    """Return (camera_dir, dataset_root) for supported dataroot shapes."""
    cam_dir = root / "samples" / camera
    if cam_dir.exists():
        return cam_dir, root

    cam_dir = root / camera
    if cam_dir.exists():
        return cam_dir, root.parent if root.name == "samples" else root

    if root.exists():
        return root, root.parent.parent if root.parent.name == "samples" else root.parent

    raise FileNotFoundError(f"Could not find camera directory for {camera} in {root}")


def _default_eval_labels_path(output_path: str) -> Path:
    out_path = Path(output_path)
    return out_path.with_name(f"{out_path.stem}_evaluation_labels.jsonl")


def _write_evaluation_labels_template(
    clips: List[Dict[str, object]],
    output_path: Path,
    include_splits: Tuple[str, ...] = ("val", "test"),
) -> int:
    rows = []
    for clip in clips:
        split = str(clip["split"])
        if split not in include_splits:
            continue
        rows.append(
            {
                "clip_id": str(clip["clip_id"]),
                "split": split,
                "scene_id": str(clip["scene_id"]),
                "camera": str(clip["camera"]),
                "binary_label": clip.get("evaluation_label"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def _scan_scene_frames(cam_dir: Path, camera: str) -> Dict[str, List[Tuple[int, Path]]]:
    scenes: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    valid_extensions = {".jpg", ".jpeg", ".png"}
    count = 0
    for entry in cam_dir.iterdir():
        if entry.suffix.lower() not in valid_extensions:
            continue
        scene_id, ts = parse_scene_and_ts(entry.name, camera)
        if scene_id:
            scenes[scene_id].append((ts, entry))
            count += 1
    logger.info(f"Found {count} frames in {len(scenes)} scenes at {cam_dir}")
    return scenes


def _build_clips(
    scenes: Dict[str, List[Tuple[int, Path]]],
    root: Path,
    dataset_root: Path,
    camera: str,
    window_size: int,
    stride: int,
    split_fn: Callable[[str], str],
    relative_paths: bool,
) -> List[Dict[str, object]]:
    clips: List[Dict[str, object]] = []
    for scene_id in sorted(scenes.keys()):
        items = scenes[scene_id]
        items.sort(key=lambda x: x[0])
        n_frames = len(items)

        for i in range(0, max(0, n_frames - window_size + 1), stride):
            chunk = items[i : i + window_size]
            if len(chunk) < window_size:
                break

            timestamps = [t for t, _ in chunk]
            clip_id = f"{scene_id}__{camera}__{timestamps[0]}__{timestamps[-1]}"

            if relative_paths:
                frame_paths = []
                for _, p in chunk:
                    try:
                        frame_paths.append(str(p.relative_to(root)))
                    except ValueError:
                        frame_paths.append(str(p.absolute()))
            else:
                frame_paths = [str(p.absolute()) for _, p in chunk]

            clips.append(
                {
                    "clip_id": clip_id,
                    "split": split_fn(scene_id),
                    "scene_id": scene_id,
                    "camera": camera,
                    "frame_paths": frame_paths,
                    "timestamps": timestamps,
                    "metadata": {"source_dataset": dataset_root.name},
                }
            )
    return clips


def _build_test_clips(
    roots: Sequence[str],
    camera: str,
    window_size: int,
    stride: int,
    evaluation_label: int,
) -> List[Dict[str, object]]:
    clips: List[Dict[str, object]] = []
    for root_str in roots:
        test_root = Path(root_str)
        test_cam_dir, test_dataset_root = _resolve_camera_dir(test_root, camera)
        logger.info(f"Scanning labeled test root {test_cam_dir}...")
        test_scenes = _scan_scene_frames(test_cam_dir, camera)
        root_clips = _build_clips(
            scenes=test_scenes,
            root=test_root,
            dataset_root=test_dataset_root,
            camera=camera,
            window_size=window_size,
            stride=stride,
            split_fn=lambda sid: "test",
            relative_paths=False,
        )
        for clip in root_clips:
            clip["evaluation_label"] = evaluation_label
        logger.info(
            "Labeled test root %s: added %d clips with binary_label=%d",
            root_str,
            len(root_clips),
            evaluation_label,
        )
        clips.extend(root_clips)
    return clips


def _sample_clips(
    clips: Sequence[Dict[str, object]],
    sample_size: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    if sample_size >= len(clips):
        return list(clips)
    sampled = rng.sample(list(clips), sample_size)
    sampled.sort(key=lambda clip: str(clip["clip_id"]))
    return sampled


def build_manifest(
    dataroot: str,
    output_path: str,
    camera: str = "CAM_FRONT",
    window_size: int = 16,
    stride: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    relative_paths: bool = True,
    evaluation_labels_output: Optional[str] = None,
    evaluation_label_splits: Tuple[str, ...] = ("val", "test"),
    positive_test_roots: Optional[Sequence[str]] = None,
    negative_test_roots: Optional[Sequence[str]] = None,
) -> int:
    """Scan directory and build clip manifest."""
    root = Path(dataroot)
    cam_dir, dataset_root = _resolve_camera_dir(root, camera)

    logger.info(f"Scanning primary root {cam_dir}...")
    scenes = _scan_scene_frames(cam_dir, camera)

    split_map = _assign_scene_splits(
        scenes.keys(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    clips = _build_clips(
        scenes=scenes,
        root=root,
        dataset_root=dataset_root,
        camera=camera,
        window_size=window_size,
        stride=stride,
        split_fn=lambda sid: split_map[sid],
        relative_paths=relative_paths,
    )

    positive_test_clips = _build_test_clips(
        positive_test_roots or (),
        camera=camera,
        window_size=window_size,
        stride=stride,
        evaluation_label=1,
    )
    negative_test_clips = _build_test_clips(
        negative_test_roots or (),
        camera=camera,
        window_size=window_size,
        stride=stride,
        evaluation_label=0,
    )

    if positive_test_clips and negative_test_clips:
        sample_size = min(len(positive_test_clips), len(negative_test_clips))
        rng = random.Random(seed)
        positive_test_clips = _sample_clips(positive_test_clips, sample_size, rng)
        negative_test_clips = _sample_clips(negative_test_clips, sample_size, rng)
        logger.info(
            "Balanced labeled test pool to %d positive and %d negative clips",
            len(positive_test_clips),
            len(negative_test_clips),
        )

    clips.extend(positive_test_clips)
    clips.extend(negative_test_clips)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for clip in clips:
            f.write(json.dumps(clip) + "\n")

    eval_labels_path = Path(evaluation_labels_output) if evaluation_labels_output else _default_eval_labels_path(output_path)
    num_eval_rows = _write_evaluation_labels_template(
        clips,
        eval_labels_path,
        include_splits=evaluation_label_splits,
    )

    logger.info(f"Wrote {len(clips)} clips to {out_path}")
    logger.info(f"Wrote {num_eval_rows} evaluation label rows to {eval_labels_path}")
    return len(clips)


def main():
    parser = argparse.ArgumentParser(description="Build a v1 JEPA clip manifest")
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes root (e.g. v1.0-mini)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera channel")
    parser.add_argument("--window", type=int, default=16, help="Frames per clip")
    parser.add_argument("--stride", type=int, default=16, help="Stride between clips")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Scene-level train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Scene-level validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument("--absolute", action="store_true", help="Store absolute paths instead of relative")
    parser.add_argument(
        "--evaluation-labels-output",
        default=None,
        help="Optional output JSONL path for binary evaluation labels template. Defaults next to the manifest.",
    )
    parser.add_argument(
        "--positive-test",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Frames root whose clips are forced into the 'test' split and labeled 1. "
            "Repeatable; absolute paths are stored so the manifest is self-contained."
        ),
    )
    parser.add_argument(
        "--negative-test",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Frames root whose clips are forced into the 'test' split and labeled 0. "
            "Repeatable; absolute paths are stored so the manifest is self-contained."
        ),
    )

    args = parser.parse_args()

    count = build_manifest(
        args.dataroot,
        args.output,
        args.camera,
        args.window,
        args.stride,
        args.train_ratio,
        args.val_ratio,
        args.seed,
        relative_paths=not args.absolute,
        evaluation_labels_output=args.evaluation_labels_output,
        positive_test_roots=args.positive_test,
        negative_test_roots=args.negative_test,
    )
    print(f"Wrote {count} clips to {args.output}")


if __name__ == "__main__":
    main()
