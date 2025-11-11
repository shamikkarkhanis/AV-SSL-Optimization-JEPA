import argparse
import json
import os
from collections import defaultdict
# Reuse filename parsing from existing manifest builder if available
try:
    from build_manifest import parse_scene_and_ts  # type: ignore
except Exception:

    def parse_scene_and_ts(filename: str, camera: str):
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


def build_clips_manifest(
    dataroot: str,
    out_path: str = "clips_manifest.jsonl",
    camera: str = "CAM_FRONT",
    window: int = 16,
    stride: int = 16,
    min_remainder: int = 0,
):
    """
    Build a JSONL manifest of fixed-length clips from raw frames.

    - Scans: <dataroot>/samples/<camera>
    - Groups frames into scenes based on filename prefix
    - Sorts by timestamp
    - Emits sliding windows of `window` frames with `stride` between starts
      (default non-overlapping: window=16, stride=16)

    Each line (per clip):
      {"scene": <scene_id>, "camera": <camera>, "frames": [paths...], "timestamps": [ints...]}

    If a scene has a remainder shorter than `window`, it is skipped unless
    `min_remainder` > 0 and remainder >= min_remainder, in which case it is
    emitted with the last frame repeated to reach `window` length.
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

    n_scenes = 0
    n_clips = 0
    with open(out_path, "w") as f:
        for scene_id, items in sorted(scenes.items()):
            items.sort(key=lambda x: x[0])
            n = len(items)
            if n == 0:
                continue
            n_scenes += 1
            # Full windows
            for i in range(0, max(0, n - window + 1), stride):
                chunk = items[i : i + window]
                if len(chunk) < window:
                    break
                frames = [p for _, p in chunk]
                tss = [t for t, _ in chunk]
                rec = {
                    "scene": scene_id,
                    "camera": camera,
                    "frames": frames,
                    "timestamps": tss,
                }
                f.write(json.dumps(rec) + "\n")
                n_clips += 1

            # Handle remainder if requested
            rem_start = (n // stride) * stride
            if rem_start < n and min_remainder > 0:
                rem = items[-min(window, n - rem_start) :]
                if len(rem) >= min_remainder:
                    # pad by repeating last frame to reach window length
                    last = rem[-1]
                    padded = rem + [last] * (window - len(rem))
                    frames = [p for _, p in padded]
                    tss = [t for t, _ in padded]
                    rec = {
                        "scene": scene_id,
                        "camera": camera,
                        "frames": frames,
                        "timestamps": tss,
                    }
                    f.write(json.dumps(rec) + "\n")
                    n_clips += 1

    print(f"Wrote {n_clips} clips across {n_scenes} scenes to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build fixed-length clips manifest (JSONL)"
    )
    ap.add_argument(
        "--dataroot", default="data/sets/nuscenes", help="Path to dataset root"
    )
    ap.add_argument(
        "--camera", default="CAM_FRONT", help="Camera channel (e.g., CAM_FRONT)"
    )
    ap.add_argument("--out", default="clips_manifest.jsonl", help="Output JSONL path")
    ap.add_argument("--window", type=int, default=16, help="Frames per clip")
    ap.add_argument("--stride", type=int, default=16, help="Stride between clip starts")
    ap.add_argument(
        "--min-remainder",
        type=int,
        default=0,
        help="If >0, emit final short clip padded to window when remainder >= this value",
    )
    args = ap.parse_args()

    build_clips_manifest(
        dataroot=args.dataroot,
        out_path=args.out,
        camera=args.camera,
        window=args.window,
        stride=args.stride,
        min_remainder=args.min_remainder,
    )
