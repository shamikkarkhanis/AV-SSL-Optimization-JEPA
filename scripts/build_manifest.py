"""Build a v1 JSONL clip manifest from a nuScenes-style data directory."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    scene_ids: Iterable[str],
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
                "binary_label": None,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


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
) -> int:
    """Scan directory and build clip manifest."""
    root = Path(dataroot)
    cam_dir, dataset_root = _resolve_camera_dir(root, camera)
        
    logger.info(f"Scanning {cam_dir}...")
    
    # Group by scene
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
            
    logger.info(f"Found {count} frames in {len(scenes)} scenes")
    split_map = _assign_scene_splits(
        scenes.keys(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    
    # Build clips
    clips = []
    
    for scene_id in sorted(scenes.keys()):
        items = scenes[scene_id]
        # Sort by timestamp
        items.sort(key=lambda x: x[0])
        
        n_frames = len(items)
        
        # Sliding window
        for i in range(0, max(0, n_frames - window_size + 1), stride):
            chunk = items[i : i + window_size]
            
            if len(chunk) < window_size:
                break
                
            timestamps = [t for t, _ in chunk]
            clip_id = f"{scene_id}__{camera}__{timestamps[0]}__{timestamps[-1]}"
            
            # Paths
            if relative_paths:
                frame_paths = []
                for _, p in chunk:
                    try:
                        rel_path = p.relative_to(root)
                        frame_paths.append(str(rel_path))
                    except ValueError:
                        frame_paths.append(str(p.absolute()))
            else:
                frame_paths = [str(p.absolute()) for _, p in chunk]
            
            record = {
                "clip_id": clip_id,
                "split": split_map[scene_id],
                "scene_id": scene_id,
                "camera": camera,
                "frame_paths": frame_paths,
                "timestamps": timestamps,
                "metadata": {
                    "source_dataset": dataset_root.name,
                },
            }
            clips.append(record)
            
    # Write output
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
    )
    print(f"Wrote {count} clips to {args.output}")


if __name__ == "__main__":
    main()
