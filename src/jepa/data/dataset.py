"""JEPA Dataset for JSONL Clip Manifests"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from PIL import Image


class JEPADataset(Dataset):
    """Dataset for loading video clips from JSONL manifests.
    
    Args:
        manifest_path: Path to JSONL manifest file
        data_root: Optional root directory to prepend to relative paths
        transform: Optional transform to apply to frames
        frames_per_clip: Number of frames per clip (default: 16)
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        transform=None,
        frames_per_clip: int = 16,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else None
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        
        # Load all manifest lines into memory
        with open(self.manifest_path, 'r') as f:
            self.lines = f.read().splitlines()
    
    def __len__(self) -> int:
        return len(self.lines)
    
    def _resolve_path(self, path: str) -> str:
        """Resolve file path, handling relative/absolute and data_root."""
        # If absolute, use as is
        if os.path.isabs(path):
            return path
            
        # If relative and we have a root, prepend it
        if self.data_root:
            return str(self.data_root / path)
            
        # If relative and no root, assume relative to CWD (fallback)
        return path
    
    def __getitem__(self, idx: int) -> Dict:
        # Parse JSONL line
        record = json.loads(self.lines[idx])
        
        # Extract frame paths
        frame_paths: List[str] = record.get("frame_paths") or record.get("frames")
        if frame_paths is None:
            raise KeyError(f"Record {idx} missing 'frame_paths' or 'frames' key")
        
        # Validate frame count
        if len(frame_paths) != self.frames_per_clip:
            raise ValueError(
                f"Expected {self.frames_per_clip} frames, got {len(frame_paths)} "
                f"in record {idx}"
            )
        
        # Resolve paths and load images
        resolved_paths = [self._resolve_path(p) for p in frame_paths]
        frames = [Image.open(p).convert("RGB") for p in resolved_paths]
        
        # Resize to 224x224
        frames = [img.resize((224, 224), Image.BICUBIC) for img in frames]
        
        # Metadata
        meta = {
            "scene": record.get("scene"),
            "camera": record.get("camera"),
            "frame_paths": resolved_paths,
        }
        
        # Apply transform
        if self.transform:
            result = self.transform(frames)
            result["meta"] = meta
            return result
        else:
            return {"frames": frames, "meta": meta}


class TubeletDataset(Dataset):
    """Dataset for tubelet-based training."""
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        tubelet_size: int = 2,
        transform=None,
    ):
        self.base_dataset = JEPADataset(
            manifest_path, 
            data_root=data_root, 
            transform=None
        )
        self.tubelet_size = tubelet_size
        self.transform = transform
        
        frames_per_clip = self.base_dataset.frames_per_clip
        self.tubelets_per_clip = frames_per_clip // tubelet_size
    
    def __len__(self) -> int:
        return len(self.base_dataset) * self.tubelets_per_clip
    
    def __getitem__(self, idx: int) -> Dict:
        clip_idx = idx // self.tubelets_per_clip
        tubelet_idx = idx % self.tubelets_per_clip
        
        clip_data = self.base_dataset[clip_idx]
        frames = clip_data["frames"]
        
        start = tubelet_idx * self.tubelet_size
        end = start + self.tubelet_size
        tubelet_frames = frames[start:end]
        
        if self.transform:
            result = self.transform(tubelet_frames)
            result["meta"] = clip_data["meta"]
            result["meta"]["tubelet_idx"] = tubelet_idx
            return result
        else:
            return {
                "frames": tubelet_frames,
                "meta": {**clip_data["meta"], "tubelet_idx": tubelet_idx},
            }
