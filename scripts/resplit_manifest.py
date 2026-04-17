"""Reassign a fraction of test scenes in a manifest into the validation split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def select_val_scenes(
    rows: Sequence[Dict[str, Any]],
    val_fraction_of_test_scenes: float,
    seed: int,
) -> Set[str]:
    if not 0.0 <= val_fraction_of_test_scenes <= 1.0:
        raise ValueError("val_fraction_of_test_scenes must be between 0.0 and 1.0.")

    test_scenes = sorted(
        {
            str(row.get("scene_id") or "")
            for row in rows
            if row.get("split") == "test" and row.get("scene_id")
        }
    )
    if not test_scenes or val_fraction_of_test_scenes == 0.0:
        return set()

    rng = random.Random(seed)
    rng.shuffle(test_scenes)
    num_val_scenes = int(len(test_scenes) * val_fraction_of_test_scenes)
    if num_val_scenes == 0 and val_fraction_of_test_scenes > 0.0:
        num_val_scenes = 1
    return set(test_scenes[:num_val_scenes])


def rewrite_rows_with_val_scenes(
    rows: Sequence[Dict[str, Any]],
    val_scenes: Set[str],
) -> Tuple[List[Dict[str, Any]], int]:
    rewritten: List[Dict[str, Any]] = []
    reassigned = 0
    for row in rows:
        updated = dict(row)
        if updated.get("split") == "test" and str(updated.get("scene_id") or "") in val_scenes:
            updated["split"] = "val"
            reassigned += 1
        rewritten.append(updated)
    return rewritten, reassigned


def resplit_manifest(
    input_path: str | Path,
    output_path: str | Path,
    val_fraction_of_test_scenes: float,
    seed: int = 42,
    labels_input_path: Optional[str | Path] = None,
    labels_output_path: Optional[str | Path] = None,
    overwrite: bool = False,
) -> Dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    rows = _read_jsonl(input_path)
    val_scenes = select_val_scenes(rows, val_fraction_of_test_scenes, seed)
    rewritten_rows, reassigned_clips = rewrite_rows_with_val_scenes(rows, val_scenes)
    _write_jsonl(output_path, rewritten_rows)

    result = {
        "manifest_rows": len(rewritten_rows),
        "reassigned_scenes": len(val_scenes),
        "reassigned_clips": reassigned_clips,
    }

    if labels_input_path is not None:
        labels_input = Path(labels_input_path)
        labels_output = Path(labels_output_path) if labels_output_path else labels_input
        if labels_output.exists() and labels_output != labels_input and not overwrite:
            raise FileExistsError(f"{labels_output} already exists. Use --overwrite to replace it.")
        label_rows = _read_jsonl(labels_input)
        rewritten_labels, reassigned_labels = rewrite_rows_with_val_scenes(label_rows, val_scenes)
        _write_jsonl(labels_output, rewritten_labels)
        result["label_rows"] = len(rewritten_labels)
        result["reassigned_label_rows"] = reassigned_labels

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move a fraction of test scenes in a manifest into the validation split."
    )
    parser.add_argument("--input", required=True, help="Input manifest JSONL path.")
    parser.add_argument("--output", required=True, help="Output manifest JSONL path.")
    parser.add_argument(
        "--val-fraction-of-test",
        type=float,
        required=True,
        help="Fraction of current test scenes to relabel as val.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument(
        "--labels-input",
        default=None,
        help="Optional evaluation-labels JSONL to rewrite with the same scene split changes.",
    )
    parser.add_argument(
        "--labels-output",
        default=None,
        help="Optional output path for rewritten labels JSONL. Defaults to overwriting --labels-input.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    args = parser.parse_args()

    result = resplit_manifest(
        input_path=args.input,
        output_path=args.output,
        val_fraction_of_test_scenes=args.val_fraction_of_test,
        seed=args.seed,
        labels_input_path=args.labels_input,
        labels_output_path=args.labels_output,
        overwrite=args.overwrite,
    )
    print(
        "Wrote {manifest_rows} manifest rows to {output}; moved {reassigned_scenes} test scenes "
        "({reassigned_clips} clips) to val.".format(
            output=args.output,
            **result,
        )
    )
    if "label_rows" in result:
        labels_output = args.labels_output or args.labels_input
        print(
            "Wrote {label_rows} label rows to {output}; updated {reassigned_label_rows} rows.".format(
                output=labels_output,
                **result,
            )
        )


if __name__ == "__main__":
    main()
