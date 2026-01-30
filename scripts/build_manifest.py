"""Build JSONL Manifest from nuScenes Data Directory"""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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


def build_manifest(
    dataroot: str,
    output_path: str,
    camera: str = "CAM_FRONT",
    window_size: int = 16,
    stride: int = 16,
    relative_paths: bool = True,
):
    """Scan directory and build clip manifest."""
    root = Path(dataroot)
    # If dataroot points to the version folder (v1.0-mini), go deeper
    # If it points to samples/CAM_FRONT, use that
    
    # Try to find the camera directory
    cam_dir = root / "samples" / camera
    if not cam_dir.exists():
        # Maybe root IS the samples dir?
        cam_dir = root / camera
        if not cam_dir.exists():
            # Maybe root IS the camera dir?
            cam_dir = root
    
    if not cam_dir.exists():
        raise FileNotFoundError(f"Could not find camera directory for {camera} in {dataroot}")
        
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
            
            # Paths
            if relative_paths:
                # Store path relative to dataroot parent (e.g. samples/CAM_FRONT/img.jpg)
                # If dataroot is 'v1.0-mini', we want 'samples/CAM_FRONT/img.jpg'
                frames = []
                for _, p in chunk:
                    try:
                        # Try to make relative to the provided dataroot
                        rel_path = p.relative_to(root)
                        frames.append(str(rel_path))
                    except ValueError:
                        # Fallback to absolute if not in root
                        frames.append(str(p.absolute()))
            else:
                frames = [str(p.absolute()) for _, p in chunk]
            
            record = {
                "scene": scene_id,
                "camera": camera,
                "frames": frames,
                "timestamps": timestamps,
            }
            clips.append(record)
            
    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        for clip in clips:
            f.write(json.dumps(clip) + "\n")
            
    logger.info(f"Wrote {len(clips)} clips to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build JEPA Clip Manifest")
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes root (e.g. v1.0-mini)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera channel")
    parser.add_argument("--window", type=int, default=16, help="Frames per clip")
    parser.add_argument("--stride", type=int, default=16, help="Stride between clips")
    parser.add_argument("--absolute", action="store_true", help="Store absolute paths instead of relative")
    
    args = parser.parse_args()
    
    build_manifest(
        args.dataroot,
        args.output,
        args.camera,
        args.window,
        args.stride,
        relative_paths=not args.absolute
    )


if __name__ == "__main__":
    main()
