"""JEPA Dataset for JSONL Clip Manifests"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from PIL import Image


class JEPADataset(Dataset):
    """Dataset for loading video clips from JSONL manifests.
    
    Compatible with nuScenes manifest format (videomae/clips_manifest.jsonl).
    Each line in the manifest is a JSON object with frame paths.
    
    Args:
        manifest_path: Path to JSONL manifest file
        transform: Optional transform to apply to frames (e.g., MaskTubelet)
        frames_per_clip: Number of frames per clip (default: 16)
    
    Returns:
        Dictionary with keys:
            - frames: List of PIL Images (if no transform)
            - clean_frames: Tensor after transform (T, C, H, W)
            - masked_frames: Tensor with masking applied
            - mask: Boolean mask array
            - meta: Metadata dict (scene, camera, frame_paths)
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        transform=None,
        frames_per_clip: int = 16,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        
        # Load all manifest lines into memory
        with open(self.manifest_path, 'r') as f:
            self.lines = f.read().splitlines()
    
    def __len__(self) -> int:
        return len(self.lines)
    
    def __getitem__(self, idx: int) -> Dict:
        # Parse JSONL line
        record = json.loads(self.lines[idx])
        
        # Extract frame paths (supports both "frames" and "frame_paths" keys)
        frame_paths: List[str] = record.get("frame_paths") or record.get("frames")
        if frame_paths is None:
            raise KeyError(f"Record {idx} missing 'frame_paths' or 'frames' key")
        
        # Validate frame count
        if len(frame_paths) != self.frames_per_clip:
            raise ValueError(
                f"Expected {self.frames_per_clip} frames, got {len(frame_paths)} "
                f"in record {idx}"
            )
        
        # Load images as PIL RGB
        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        
        # Resize to 224x224 (expected by VJEPA encoder)
        frames = [img.resize((224, 224), Image.BICUBIC) for img in frames]
        
        # Metadata
        meta = {
            "scene": record.get("scene"),
            "camera": record.get("camera"),
            "frame_paths": frame_paths,
        }
        
        # Apply transform if provided
        if self.transform:
            result = self.transform(frames)
            result["meta"] = meta
            return result
        else:
            return {"frames": frames, "meta": meta}


class TubeletDataset(Dataset):
    """Dataset for tubelet-based training (groups frames into tubelets).
    
    Converts 16-frame clips into tubelets (groups of `tubelet_size` frames).
    Used for JEPA training where encoder processes 2-frame tubelets.
    
    Args:
        manifest_path: Path to JSONL manifest
        tubelet_size: Number of frames per tubelet (default: 2)
        transform: Transform to apply (e.g., MaskTubelet)
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        tubelet_size: int = 2,
        transform=None,
    ):
        self.base_dataset = JEPADataset(manifest_path, transform=None)
        self.tubelet_size = tubelet_size
        self.transform = transform
        
        # Calculate number of tubelets per clip
        frames_per_clip = self.base_dataset.frames_per_clip
        self.tubelets_per_clip = frames_per_clip // tubelet_size
    
    def __len__(self) -> int:
        return len(self.base_dataset) * self.tubelets_per_clip
    
    def __getitem__(self, idx: int) -> Dict:
        # Map global index to (clip_idx, tubelet_idx)
        clip_idx = idx // self.tubelets_per_clip
        tubelet_idx = idx % self.tubelets_per_clip
        
        # Get full clip
        clip_data = self.base_dataset[clip_idx]
        frames = clip_data["frames"]
        
        # Extract tubelet frames
        start = tubelet_idx * self.tubelet_size
        end = start + self.tubelet_size
        tubelet_frames = frames[start:end]
        
        # Apply transform
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
