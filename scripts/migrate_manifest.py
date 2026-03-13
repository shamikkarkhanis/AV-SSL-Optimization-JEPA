"""Migrate legacy clip manifests to the v1 clip schema."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _stable_clip_id(record: Dict[str, Any], clip_index: int) -> str:
    scene_id = str(record.get("scene_id") or record.get("scene") or "unknown-scene")
    camera = str(record.get("camera") or "unknown-camera")
    timestamps = list(record.get("timestamps") or [])
    if timestamps:
        start_ts = timestamps[0]
        end_ts = timestamps[-1]
        return f"{scene_id}__{camera}__{start_ts}__{end_ts}"
    return f"{scene_id}__{camera}__clip_{clip_index:06d}"


def _assign_scene_splits(
    scene_ids: Iterable[str],
    train_ratio: float,
    score_ratio: float,
    seed: int,
) -> Dict[str, str]:
    unique_scenes = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_scenes)

    num_scenes = len(unique_scenes)
    train_cut = int(num_scenes * train_ratio)
    score_cut = train_cut + int(num_scenes * score_ratio)

    split_map: Dict[str, str] = {}
    for index, scene_id in enumerate(unique_scenes):
        if index < train_cut:
            split_map[scene_id] = "train"
        elif index < score_cut:
            split_map[scene_id] = "val"
        else:
            split_map[scene_id] = "test"
    return split_map


def migrate_manifest(
    input_path: str | Path,
    output_path: str | Path,
    train_ratio: float = 0.7,
    score_ratio: float = 0.15,
    seed: int = 42,
    overwrite: bool = False,
) -> int:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    raw_records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_records.append(json.loads(line))

    split_map = _assign_scene_splits(
        (str(record.get("scene_id") or record.get("scene") or "unknown-scene") for record in raw_records),
        train_ratio=train_ratio,
        score_ratio=score_ratio,
        seed=seed,
    )

    migrated: List[Dict[str, Any]] = []
    for clip_index, record in enumerate(raw_records):
        scene_id = str(record.get("scene_id") or record.get("scene") or "unknown-scene")
        frame_paths = list(record.get("frame_paths") or record.get("frames") or [])
        migrated.append(
            {
                "clip_id": str(record.get("clip_id") or _stable_clip_id(record, clip_index)),
                "split": str(record.get("split") or split_map[scene_id]),
                "scene_id": scene_id,
                "camera": record.get("camera"),
                "frame_paths": frame_paths,
                "timestamps": list(record.get("timestamps") or []),
                "metadata": {
                    "source_dataset": record.get("source_dataset"),
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in migrated:
            handle.write(json.dumps(record) + "\n")
    return len(migrated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a legacy clip manifest to the v1 schema.")
    parser.add_argument("--input", required=True, help="Legacy JSONL manifest path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Scene-level train split ratio.")
    parser.add_argument("--score-ratio", type=float, default=0.15, help="Scene-level validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic scene split assignment.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    args = parser.parse_args()

    migrated = migrate_manifest(
        input_path=args.input,
        output_path=args.output,
        train_ratio=args.train_ratio,
        score_ratio=args.score_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Migrated {migrated} clips to {args.output}")


if __name__ == "__main__":
    main()
