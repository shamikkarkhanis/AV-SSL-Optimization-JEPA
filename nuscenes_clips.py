# NuScenes handler
from typing import List
import json
from PIL import Image

class NuScenesClips():
    """
    Each line of the JSONL manifest must contain:
      {
        "frame_paths": [".../img_000.jpg", "...", "... (16 total)"],
        ... (optional metadata: scene_id, camera, timestamps, etc.)
      }
    """
    def __init__(self, manifest_path: str, frames_per_clip: int = 16):
        self.manifest_path = manifest_path
        with open(manifest_path, "r") as f:
            self.lines = f.read().splitlines()
        self.frames_per_clip = frames_per_clip

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        rec = json.loads(self.lines[idx])
        # Support either key name used by different builders
        paths: List[str] = rec.get("frame_paths") or rec.get("frames")
        if paths is None:
            raise KeyError("Record missing 'frame_paths' or 'frames'")
        assert len(paths) == self.frames_per_clip, f"Need {self.frames_per_clip} frames, got {len(paths)}"
        frames = [Image.open(p).convert("RGB") for p in paths]
        return frames  # list of 16 PIL images


if __name__ == "__main__":
    import argparse
    import numpy as np

    ap = argparse.ArgumentParser(description="Quick check: load a clip and print pixel values")
    ap.add_argument("manifest", help="Path to clips manifest JSONL")
    ap.add_argument("--idx", type=int, default=0, help="Index of clip to inspect")
    args = ap.parse_args()

    ds = NuScenesClips(args.manifest)
    frames = ds[args.idx]
    print(f"Loaded clip #{args.idx} with {len(frames)} frames")
    for i, im in enumerate(frames):
        arr = np.array(im)  # HxWx3 uint8
        print(f"Frame {i}: shape={arr.shape}, dtype={arr.dtype}")
        # Print all pixel values; beware this can be very large
        # If this is too verbose, consider slicing like arr[0:2,0:4]
        np.set_printoptions(threshold=arr.size, linewidth=120)
        print(arr)
