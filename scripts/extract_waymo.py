"""Extract FRONT-camera keyframes from Waymo TFRecords to nuScenes-compatible jpegs.

Designed to be run inside a linux/amd64 container where
`waymo-open-dataset-tf-2-11-0` has working wheels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from waymo_open_dataset import dataset_pb2


def segment_id_from_path(tfrecord_path: Path) -> str:
    stem = tfrecord_path.stem
    if stem.startswith("segment-"):
        stem = stem[len("segment-"):]
    if stem.endswith("_with_camera_labels"):
        stem = stem[: -len("_with_camera_labels")]
    return stem


def extract_one(tfrecord_path: Path, out_dir: Path, subsample: int) -> dict:
    segment_id = segment_id_from_path(tfrecord_path)
    extracted = 0
    total = 0
    stats = {"time_of_day": None, "weather": None, "location": None}
    front = dataset_pb2.CameraName.FRONT

    for data in tf.data.TFRecordDataset(str(tfrecord_path), compression_type=""):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))

        if total == 0:
            stats["time_of_day"] = frame.context.stats.time_of_day
            stats["weather"] = frame.context.stats.weather
            stats["location"] = frame.context.stats.location

        if total % subsample == 0:
            for image in frame.images:
                if image.name == front:
                    ts = frame.timestamp_micros
                    name = f"waymo-{segment_id}__CAM_FRONT__{ts}.jpg"
                    (out_dir / name).write_bytes(image.image)
                    extracted += 1
                    break
        total += 1

    return {
        "segment_id": segment_id,
        "total_frames": total,
        "extracted": extracted,
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-dir", default="data/raw/test/_staging/waymo")
    parser.add_argument("--output-dir", default="data/raw/test/waymo/samples/CAM_FRONT")
    parser.add_argument("--subsample", type=int, default=5,
                        help="Keep every Nth frame (Waymo is 10Hz; 5 -> 2Hz to match nuScenes samples).")
    args = parser.parse_args()

    staging = Path(args.staging_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfrecords = sorted(staging.glob("*.tfrecord"))
    print(f"Found {len(tfrecords)} TFRecords in {staging}\n")

    results = []
    for tr in tfrecords:
        print(f"Extracting {tr.name}...")
        r = extract_one(tr, out_dir, args.subsample)
        results.append(r)
        print(
            f"  segment: {r['segment_id']}\n"
            f"  frames: {r['extracted']} / {r['total_frames']}  "
            f"({r['time_of_day']} / {r['weather']} / {r['location']})"
        )

    print("\n=== Summary ===")
    total_out = sum(r["extracted"] for r in results)
    print(f"Extracted {total_out} FRONT-camera keyframes into {out_dir}")
    print()
    print(f"{'segment_id':<45} {'time':<10} {'weather':<10} {'location':<20} frames")
    for r in results:
        print(f"{r['segment_id']:<45} {r['time_of_day']:<10} {r['weather']:<10} "
              f"{r['location']:<20} {r['extracted']}")


if __name__ == "__main__":
    main()
