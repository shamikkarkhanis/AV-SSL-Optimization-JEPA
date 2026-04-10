"""Build a v1 clip manifest from extracted video frames."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _frame_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return (int(digits), stem)
    return (0, stem)


def _assign_scene_splits(
    scene_ids: Iterable[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    unique_scenes = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_scenes)

    n_scenes = len(unique_scenes)
    train_cut = int(n_scenes * train_ratio)
    val_cut = train_cut + int(n_scenes * val_ratio)

    split_map: Dict[str, str] = {}
    for idx, scene_id in enumerate(unique_scenes):
        if idx < train_cut:
            split_map[scene_id] = "train"
        elif idx < val_cut:
            split_map[scene_id] = "val"
        else:
            split_map[scene_id] = "test"
    return split_map


def _default_eval_labels_path(output_path: str | Path) -> Path:
    out_path = Path(output_path)
    return out_path.with_name(f"{out_path.stem}_evaluation_labels.jsonl")


def _write_evaluation_labels_template(
    records: Sequence[Dict[str, object]],
    output_path: Path,
    include_splits: Tuple[str, ...] = ("val", "test"),
) -> int:
    rows = []
    for record in records:
        split = str(record["split"])
        if split not in include_splits:
            continue
        rows.append(
            {
                "clip_id": str(record["clip_id"]),
                "split": split,
                "scene_id": str(record["scene_id"]),
                "camera": str(record["camera"]),
                "binary_label": None,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


def _window_records(
    root: Path,
    scene_id: str,
    camera: str,
    frame_paths: Sequence[Path],
    clip_length: int,
    stride: int,
    split: str,
    relative_paths: bool,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if len(frame_paths) < clip_length:
        return records

    for start_idx in range(0, len(frame_paths) - clip_length + 1, stride):
        chunk = list(frame_paths[start_idx : start_idx + clip_length])
        rel_or_abs = [
            str(path.relative_to(root)) if relative_paths else str(path.resolve())
            for path in chunk
        ]
        timestamps = [path.stem for path in chunk]
        clip_id = f"{scene_id}__{camera}__{chunk[0].stem}__{chunk[-1].stem}"
        records.append(
            {
                "clip_id": clip_id,
                "split": split,
                "scene_id": scene_id,
                "camera": camera,
                "frame_paths": rel_or_abs,
                "timestamps": timestamps,
                "metadata": {},
            }
        )
    return records


def build_manifest_from_frames(
    frames_root: str | Path,
    output_path: str | Path,
    clip_length: int = 16,
    stride: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    relative_paths: bool = True,
    camera_filter: str | None = None,
    evaluation_labels_output: Optional[str | Path] = None,
    evaluation_label_splits: Tuple[str, ...] = ("val", "test"),
) -> int:
    root = Path(frames_root)
    output = Path(output_path)

    scene_dirs = [path for path in root.iterdir() if path.is_dir()]
    split_map = _assign_scene_splits(
        (scene_dir.name for scene_dir in scene_dirs),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    records: List[Dict[str, object]] = []
    for scene_dir in sorted(scene_dirs):
        camera_dirs = [path for path in scene_dir.iterdir() if path.is_dir()]
        for camera_dir in sorted(camera_dirs):
            camera = camera_dir.name
            if camera_filter and camera != camera_filter:
                continue
            frame_paths = sorted(
                [path for path in camera_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS],
                key=_frame_sort_key,
            )
            records.extend(
                _window_records(
                    root=root,
                    scene_id=scene_dir.name,
                    camera=camera,
                    frame_paths=frame_paths,
                    clip_length=clip_length,
                    stride=stride,
                    split=split_map[scene_dir.name],
                    relative_paths=relative_paths,
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    eval_labels_path = (
        Path(evaluation_labels_output)
        if evaluation_labels_output
        else _default_eval_labels_path(output_path)
    )
    num_eval_rows = _write_evaluation_labels_template(
        records,
        eval_labels_path,
        include_splits=evaluation_label_splits,
    )
    print(f"Wrote {num_eval_rows} evaluation label rows to {eval_labels_path}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a v1 clip manifest from extracted frames.")
    parser.add_argument("--frames-root", required=True, help="Root directory in scene/camera/frame.jpg layout.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--clip-length", type=int, default=16, help="Frames per clip.")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Scene-level train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Scene-level validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--camera", default=None, help="Optional camera directory to include.")
    parser.add_argument("--absolute", action="store_true", help="Write absolute frame paths instead of paths relative to frames-root.")
    parser.add_argument(
        "--evaluation-labels-output",
        default=None,
        help="Optional output JSONL path for binary evaluation labels template. Defaults next to the manifest.",
    )
    args = parser.parse_args()

    count = build_manifest_from_frames(
        frames_root=args.frames_root,
        output_path=args.output,
        clip_length=args.clip_length,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        relative_paths=not args.absolute,
        camera_filter=args.camera,
        evaluation_labels_output=args.evaluation_labels_output,
    )
    print(f"Wrote {count} clips to {args.output}")


if __name__ == "__main__":
    main()
