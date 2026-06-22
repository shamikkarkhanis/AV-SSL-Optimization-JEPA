"""Build a randomly-sampled nuScenes clip pool for human interestingness labeling.

Produces 8-second clips (16 frames at ~2 Hz, subsampled from 12 Hz sweeps) across
all available nuScenes scenes EXCEPT held-out benchmark scenes, then writes a
manifest + an empty label template (binary_label=null) for review_labels.py.

Random sampling (not description-seeded) so the resulting labels give an unbiased
positive rate and an honest test set. Scene-disjoint splitting happens later at
train time.

Usage (on the cluster):
    python scripts/build_labeling_pool.py \
        --sweeps-root /gpfs/u/barn/EFCM/shared/data/raw/v1.0-trainval01/sweeps \
        --sweeps-root /gpfs/u/barn/EFCM/shared/data/raw/v1.0-trainval02/sweeps \
        --sweeps-root /gpfs/u/barn/EFCM/shared/data/raw/v1.0-trainval03/sweeps \
        --sweeps-root /gpfs/u/barn/EFCM/shared/data/raw/v1.0-trainval04/sweeps \
        --exclude-scene n008-2018-08-01-15-16-36-0400 ... \
        --target 400 --output-dir /gpfs/u/scratch/EFCM/EFCMdvlr/label_pool
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

CAMERA = "CAM_FRONT"


def _scan(sweeps_root: Path) -> Dict[str, List[Tuple[int, Path]]]:
    cam = sweeps_root / CAMERA
    scenes: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for entry in cam.iterdir():
        if entry.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        name = entry.stem
        marker = f"__{CAMERA}__"
        if marker not in name:
            continue
        left, right = name.split(marker, 1)
        try:
            ts = int(right)
        except ValueError:
            continue
        scenes[left].append((ts, entry))
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a random clip pool for labeling.")
    ap.add_argument("--sweeps-root", action="append", required=True, help="Repeatable sweeps root.")
    ap.add_argument("--exclude-scene", action="append", default=[], help="Scene ids to hold out.")
    ap.add_argument("--frames-per-clip", type=int, default=16)
    ap.add_argument("--step", type=int, default=6, help="Subsample stride over 12 Hz sweeps (~2 Hz at 6).")
    ap.add_argument("--max-per-scene", type=int, default=16, help="Cap clips per scene for diversity.")
    ap.add_argument("--target", type=int, default=400, help="Approx total clips to sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    exclude = set(args.exclude_scene)

    # Gather frames per scene across all roots.
    scene_frames: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for root in args.sweeps_root:
        for scene, items in _scan(Path(root)).items():
            if scene in exclude:
                continue
            scene_frames[scene].extend(items)

    span = args.frames_per_clip * args.step  # raw sweep frames consumed per clip
    all_clips: List[dict] = []
    for scene in sorted(scene_frames):
        items = sorted(scene_frames[scene], key=lambda x: x[0])
        # Non-overlapping 8s windows; subsample every `step` frame.
        windows = []
        for start in range(0, len(items) - span + 1, span):
            chunk = items[start : start + span : args.step][: args.frames_per_clip]
            if len(chunk) == args.frames_per_clip:
                windows.append(chunk)
        rng.shuffle(windows)
        for chunk in windows[: args.max_per_scene]:
            ts = [t for t, _ in chunk]
            clip_id = f"{scene}__{CAMERA}__{ts[0]}__{ts[-1]}"
            all_clips.append({
                "clip_id": clip_id,
                "split": "label",
                "scene_id": scene,
                "camera": CAMERA,
                "frame_paths": [str(p.absolute()) for _, p in chunk],
                "timestamps": ts,
                "metadata": {"source_dataset": "nuscenes_label_pool"},
            })

    rng.shuffle(all_clips)
    pool = all_clips[: args.target]
    pool.sort(key=lambda c: c["clip_id"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "label_pool.jsonl"
    lab_path = out / "label_pool_evaluation_labels.jsonl"
    with open(man_path, "w") as mf, open(lab_path, "w") as lf:
        for c in pool:
            mf.write(json.dumps(c) + "\n")
            lf.write(json.dumps({
                "clip_id": c["clip_id"], "split": "label",
                "scene_id": c["scene_id"], "camera": CAMERA, "binary_label": None,
            }) + "\n")

    n_scenes = len({c["scene_id"] for c in pool})
    print(f"Pool: {len(pool)} clips across {n_scenes} scenes "
          f"(from {len(scene_frames)} available, {len(all_clips)} candidate clips)")
    print(f"  manifest: {man_path}")
    print(f"  labels:   {lab_path}")


if __name__ == "__main__":
    main()
