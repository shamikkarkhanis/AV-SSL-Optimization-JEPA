"""Dataset utilities for JSONL clip manifests and evaluation labels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
import torch
from torch.utils.data import Dataset


def derive_clip_id(record: Dict[str, Any]) -> str:
    """Build a stable fallback clip id for backward-compatible manifests."""
    if record.get("clip_id"):
        return str(record["clip_id"])

    payload = {
        "frame_paths": record.get("frame_paths") or record.get("frames") or [],
        "timestamps": record.get("timestamps") or [],
        "scene_id": record.get("scene_id") or record.get("scene"),
        "camera": record.get("camera"),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"derived-{digest[:16]}"


def normalize_clip_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize legacy and v1 clip manifest records to one schema."""
    frame_paths = record.get("frame_paths") or record.get("frames")
    if frame_paths is None:
        raise KeyError("Manifest record missing 'frame_paths' or 'frames'.")

    normalized = dict(record)
    normalized["clip_id"] = derive_clip_id(record)
    normalized["split"] = (
        record.get("split")
        or record.get("subset")
        or record.get("partition")
        or "unspecified"
    )
    normalized["scene_id"] = record.get("scene_id") or record.get("scene")
    normalized["frame_paths"] = list(frame_paths)
    normalized["timestamps"] = list(record.get("timestamps") or [])
    return normalized


def load_clip_manifest(
    manifest_path: Union[str, Path],
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load a JSONL clip manifest, optionally filtering by split."""
    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = normalize_clip_record(json.loads(line))
            if split is None or record["split"] == split:
                records.append(record)
    return records


def load_evaluation_labels(labels_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load normalized review-value labels from a JSONL or JSON file."""
    path = Path(labels_path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_records = payload if isinstance(payload, list) else payload["labels"]
    else:
        with open(path, "r", encoding="utf-8") as handle:
            raw_records = [json.loads(line) for line in handle if line.strip()]

    labels: List[Dict[str, Any]] = []
    for record in raw_records:
        if "clip_id" not in record:
            raise KeyError("Evaluation label record missing 'clip_id'.")
        labels.append(
            {
                "clip_id": str(record["clip_id"]),
                "review_value": record.get("review_value"),
                "review_value_grade": record.get("review_value_grade"),
                "binary_label": record.get("binary_label"),
                "reason_codes": list(record.get("reason_codes") or []),
                "reviewer_id": record.get("reviewer_id"),
                "adjudicated_label": record.get("adjudicated_label"),
                "agreement": record.get("agreement"),
            }
        )
    return labels


class JEPADataset(Dataset):
    """Dataset for loading video clips from normalized JSONL manifests."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        transform=None,
        frames_per_clip: int = 16,
        split: Optional[str] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root) if data_root else None
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.records = load_clip_manifest(self.manifest_path, split=split)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        if self.data_root:
            return str(self.data_root / path)
        return path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        frame_paths = record["frame_paths"]
        if len(frame_paths) != self.frames_per_clip:
            raise ValueError(
                f"Expected {self.frames_per_clip} frames, got {len(frame_paths)} "
                f"for clip_id={record['clip_id']}"
            )

        resolved_paths = [self._resolve_path(path) for path in frame_paths]
        frames = [Image.open(path).convert("RGB") for path in resolved_paths]
        frames = [image.resize((224, 224), Image.BICUBIC) for image in frames]

        meta = {
            "clip_id": record["clip_id"],
            "split": record["split"],
            "scene_id": record.get("scene_id") or "",
            "camera": record.get("camera") or "",
            "frame_paths": resolved_paths,
            "timestamps": record.get("timestamps") or [],
            "metadata": record.get("metadata") or {},
        }

        if self.transform:
            result = self.transform(frames)
            result["meta"] = meta
            return result
        return {"frames": frames, "meta": meta}


class TubeletDataset(Dataset):
    """Dataset that splits each clip into fixed-size tubelets."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        tubelet_size: int = 2,
        transform=None,
        split: Optional[str] = None,
        frames_per_clip: int = 16,
    ):
        self.base_dataset = JEPADataset(
            manifest_path,
            data_root=data_root,
            transform=None,
            frames_per_clip=frames_per_clip,
            split=split,
        )
        self.tubelet_size = tubelet_size
        self.transform = transform

        if self.base_dataset.frames_per_clip % tubelet_size != 0:
            raise ValueError("frames_per_clip must be divisible by tubelet_size.")
        self.tubelets_per_clip = self.base_dataset.frames_per_clip // tubelet_size

    def __len__(self) -> int:
        return len(self.base_dataset) * self.tubelets_per_clip

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip_idx = idx // self.tubelets_per_clip
        tubelet_idx = idx % self.tubelets_per_clip

        clip_data = self.base_dataset[clip_idx]
        frames = clip_data["frames"]

        start = tubelet_idx * self.tubelet_size
        end = start + self.tubelet_size
        tubelet_frames = frames[start:end]

        meta = dict(clip_data["meta"])
        meta["tubelet_idx"] = tubelet_idx

        if self.transform:
            result = self.transform(tubelet_frames)
            result["meta"] = meta
            return result
        return {"frames": tubelet_frames, "meta": meta}
