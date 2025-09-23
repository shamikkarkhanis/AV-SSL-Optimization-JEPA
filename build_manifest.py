import os
import json
import argparse
from collections import defaultdict


def parse_scene_and_ts(filename: str, camera: str):
    """
    Parse a NuScenes sample image filename to extract scene id and timestamp.
    Expected pattern: <scene_prefix>__<CAMERA>__<timestamp>.jpg
    Example: n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg
    Returns (scene_id, timestamp_int) or (None, None) if not parseable.
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    marker = f"__{camera}__"
    if marker not in name:
        return None, None
    left, right = name.split(marker, 1)
    try:
        ts = int(right)
    except ValueError:
        return None, None
    return left, ts


def build_manifest(dataroot: str, out_path: str, camera: str = "CAM_FRONT"):
    """
    Scan dataroot/samples/<camera>, group images into scenes by filename prefix,
    and write a JSONL manifest with one record per scene:

      {"scene": <scene_id>, "camera": <camera>, "images": [<abs_or_rel_paths_sorted_by_time>]}
    """
    cam_dir = os.path.join(dataroot, "samples", camera)
    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")

    scenes = defaultdict(list)
    for entry in os.listdir(cam_dir):
        if not entry.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(cam_dir, entry)
        scene_id, ts = parse_scene_and_ts(entry, camera)
        if scene_id is None:
            continue
        scenes[scene_id].append((ts, path))

    # Sort images within each scene by timestamp and serialize paths
    total_images = 0
    with open(out_path, "w") as f:
        for scene_id, items in sorted(scenes.items()):
            items.sort(key=lambda x: x[0])
            images = [p for _, p in items]
            total_images += len(images)
            rec = {"scene": scene_id, "camera": camera, "images": images}
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(scenes)} scenes ({total_images} images) to {out_path}")


def build_train_manifest(
    dataroot: str,
    out_train_path: str,
    out_eval_path: str,
    camera: str = "CAM_FRONT",
    window: int = 10,
    stride: int = 1,
    train_ratio: float = 0.9,
):
    """
    Build a windowed manifest JSONL with 90/10 scene-level split.

    - Groups images in samples/<camera> into scenes by filename prefix.
    - Splits scenes: first 90% train, last 10% eval (by sorted scene id).
    - Emits sliding windows of `window` frames with `stride` within each scene.

    Output row format (per window) — written to separate files:
      {"scene": <scene_id>, "camera": <camera>, "frames": [paths...], "timestamps": [ints...]}
    """
    cam_dir = os.path.join(dataroot, "samples", camera)
    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"Camera directory not found: {cam_dir}")

    scenes = defaultdict(list)
    for entry in os.listdir(cam_dir):
        if not entry.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(cam_dir, entry)
        scene_id, ts = parse_scene_and_ts(entry, camera)
        if scene_id is None:
            continue
        scenes[scene_id].append((ts, path))

    # Sort scene ids and contents
    scene_ids = sorted(scenes.keys())
    for sid in scene_ids:
        scenes[sid].sort(key=lambda x: x[0])

    # Split by scene
    split_idx = int(len(scene_ids) * train_ratio)
    train_ids = scene_ids[:split_idx]
    eval_ids = scene_ids[split_idx:]

    n_train_windows = 0
    n_eval_windows = 0
    train_counts = defaultdict(int)
    eval_counts = defaultdict(int)
    with open(out_train_path, "w") as f_train:
        # Train windows
        for sid in train_ids:
            items = scenes[sid]
            if len(items) < window:
                continue
            for i in range(0, len(items) - window + 1, stride):
                chunk = items[i : i + window]
                frames = [p for _, p in chunk]
                tss = [t for t, _ in chunk]
                rec = {"scene": sid, "camera": camera, "frames": frames, "timestamps": tss}
                f_train.write(json.dumps(rec) + "\n")
                n_train_windows += 1
                train_counts[sid] += 1

    with open(out_eval_path, "w") as f_eval:
        # Eval windows
        for sid in eval_ids:
            items = scenes[sid]
            if len(items) < window:
                continue
            for i in range(0, len(items) - window + 1, stride):
                chunk = items[i : i + window]
                frames = [p for _, p in chunk]
                tss = [t for t, _ in chunk]
                rec = {"scene": sid, "camera": camera, "frames": frames, "timestamps": tss}
                f_eval.write(json.dumps(rec) + "\n")
                n_eval_windows += 1
                eval_counts[sid] += 1

    print(
        f"Wrote windowed manifests: train_file={out_train_path} ({n_train_windows} windows, {len(train_ids)} scenes), eval_file={out_eval_path} ({n_eval_windows} windows, {len(eval_ids)} scenes)"
    )
    # Basic stats per split
    def _summarize(counts: dict):
        if not counts:
            return (0, 0.0, 0)
        vals = list(counts.values())
        return (min(vals), sum(vals) / len(vals), max(vals))

    tr_min, tr_avg, tr_max = _summarize(train_counts)
    ev_min, ev_avg, ev_max = _summarize(eval_counts)
    print(
        "Stats — train windows/scene: min={:.0f}, avg={:.2f}, max={:.0f}; eval windows/scene: min={:.0f}, avg={:.2f}, max={:.0f}".format(
            tr_min, tr_avg, tr_max, ev_min, ev_avg, ev_max
        )
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build NuScenes CAM image manifests.")
    ap.add_argument("--dataroot", default="data/sets/nuscenes", help="Path to NuScenes dataroot")
    ap.add_argument("--camera", default="CAM_FRONT", help="Camera channel to index (e.g., CAM_FRONT)")
    ap.add_argument("--out", default="scene_manifest.jsonl", help="Scene manifest JSONL path")
    ap.add_argument("--train-out", default="train_windows.jsonl", help="Train windows JSONL path")
    ap.add_argument("--eval-out", default="eval_windows.jsonl", help="Eval windows JSONL path")
    ap.add_argument("--make-windows", action="store_true", help="Also write windowed train/eval manifest")
    ap.add_argument("--window", type=int, default=10, help="Frames per window")
    ap.add_argument("--stride", type=int, default=1, help="Stride for sliding windows")
    ap.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio by scenes")
    args = ap.parse_args()

    # Always write scene-level manifest
    build_manifest(args.dataroot, args.out, args.camera)

    # Always write windowed train/eval manifests as requested
    build_train_manifest(
        dataroot=args.dataroot,
        out_train_path=args.train_out,
        out_eval_path=args.eval_out,
        camera=args.camera,
        window=args.window,
        stride=args.stride,
        train_ratio=args.train_ratio,
    )
