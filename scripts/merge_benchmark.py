"""Merge labeled per-source manifests into one fixed evaluation benchmark.

The FTC experiment cycle scores every run on ONE held-out benchmark of labeled
clips (nuScenes + Waymo + BDD100K). This stitches the existing labeled manifests
and label files into a single self-contained manifest (absolute frame paths, so
it is independent of dataset.data_root) plus a single labels file.

The label files are the source of truth for benchmark membership: only clips that
appear in a labels file (with a non-null binary_label) are included.

Usage (run on the cluster where the frames live, or anywhere -- it never opens
image files, only rewrites paths):

    python scripts/merge_benchmark.py \
        --manifest data/manifests/nuscenes_trainval.jsonl --root /path/to/v1.0-trainval \
        --manifest data/manifests/waymo_test.jsonl \
        --manifest data/manifests/bdd100k_test.jsonl \
        --labels data/manifests/nuscenes_trainval_evaluation_labels.jsonl \
        --labels data/manifests/waymo_test_evaluation_labels.jsonl \
        --labels data/manifests/bdd100k_test_evaluation_labels.jsonl \
        --output-manifest data/manifests/benchmark_test.jsonl \
        --output-labels data/manifests/benchmark_test_evaluation_labels.jsonl

--root absolutizes RELATIVE frame paths in the manifest that precedes it; already
absolute paths (e.g. Waymo/BDD injected via --test-frames-root) pass through.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _absolutize(record: Dict[str, Any], root: Optional[str]) -> Dict[str, Any]:
    out = dict(record)
    frame_paths = []
    for p in record.get("frame_paths", []):
        if os.path.isabs(p) or root is None:
            frame_paths.append(p)
        else:
            frame_paths.append(str(Path(root) / p))
    out["frame_paths"] = frame_paths
    out["split"] = "test"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge labeled manifests into one benchmark.")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        nargs="+",
        metavar=("MANIFEST", "ROOT"),
        help="Manifest path, optionally followed by a frames root to absolutize relative paths.",
    )
    parser.add_argument("--labels", action="append", required=True, help="Label JSONL (repeatable).")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument(
        "--require-label",
        action="store_true",
        help="Only include clips whose binary_label is non-null.",
    )
    args = parser.parse_args()

    # clip_id -> record (later manifests win on duplicate ids).
    clip_records: Dict[str, Dict[str, Any]] = {}
    for entry in args.manifest:
        manifest_path = Path(entry[0])
        root = entry[1] if len(entry) > 1 else None
        for record in _load_jsonl(manifest_path):
            clip_id = str(record.get("clip_id"))
            clip_records[clip_id] = _absolutize(record, root)

    # Collect labels (the membership source of truth).
    merged_labels: Dict[str, Dict[str, Any]] = {}
    for labels_path in args.labels:
        for label in _load_jsonl(Path(labels_path)):
            clip_id = str(label.get("clip_id"))
            if args.require_label and label.get("binary_label") is None:
                continue
            merged_labels[clip_id] = label

    out_manifest = Path(args.output_manifest)
    out_labels = Path(args.output_labels)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_labels.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = []
    pos = neg = 0
    with open(out_manifest, "w") as mf, open(out_labels, "w") as lf:
        for clip_id, label in merged_labels.items():
            if clip_id not in clip_records:
                missing.append(clip_id)
                continue
            mf.write(json.dumps(clip_records[clip_id]) + "\n")
            label = dict(label)
            label["split"] = "test"
            lf.write(json.dumps(label) + "\n")
            written += 1
            if label.get("binary_label") in (1, "1", True):
                pos += 1
            elif label.get("binary_label") in (0, "0", False):
                neg += 1

    print(f"Wrote {written} benchmark clips -> {out_manifest}")
    print(f"  positives={pos} negatives={neg} unlabeled/other={written - pos - neg}")
    print(f"Wrote {written} labels -> {out_labels}")
    if missing:
        print(f"WARNING: {len(missing)} labeled clip_ids had no manifest record (skipped):")
        for clip_id in missing[:10]:
            print(f"  - {clip_id}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
