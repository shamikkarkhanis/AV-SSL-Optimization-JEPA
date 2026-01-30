import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_scores(path: str) -> List[float]:
    scores: List[float] = []
    if not os.path.isfile(path):
        return scores
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                val = float(rec.get("score"))
                scores.append(val)
            except Exception:
                continue
    return scores


def _histogram(values: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    # Use numpy to compute histogram bins and counts
    counts, edges = np.histogram(values, bins=bins)
    return counts.astype(np.int64), edges


def _format_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def plot_distribution(
    scores_path: str,
    out_path: str = "renders/score_distribution.png",
    bins: int = 40,
    width: int = 900,
    height: int = 520,
) -> dict:
    """
    Load per-clip scores from a JSONL and render a histogram to `out_path`.
    Returns a dict of summary stats.
    """
    scores = _load_scores(scores_path)
    values = np.array(scores, dtype=np.float32)

    # Prepare canvas
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_l, margin_r, margin_t, margin_b = 70, 20, 40, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x0, y0 = margin_l, margin_t
    x1, y1 = x0 + plot_w, y0 + plot_h

    # Axes
    draw.rectangle([x0, y0, x1, y1], outline=(220, 220, 220), width=1)

    if values.size == 0:
        msg = "No scores found"
        tw, th = draw.textlength(msg), 14
        draw.text((width/2 - tw/2, height/2 - th/2), msg, fill=(0, 0, 0))
        img.save(out_path)
        return {}

    counts, edges = _histogram(values, bins=bins)
    max_count = max(1, int(counts.max()))

    # Bars
    bar_w = plot_w / bins
    for i in range(bins):
        c = int(counts[i])
        if c <= 0:
            continue
        # bar rectangle
        bx0 = int(x0 + i * bar_w + 1)
        bx1 = int(x0 + (i + 1) * bar_w - 1)
        bh = int(plot_h * (c / max_count))
        by0 = y1 - bh
        draw.rectangle([bx0, by0, bx1, y1], fill=(66, 135, 245))

    # X ticks (min, median, max)
    ticks = [float(edges[0]), float(np.median(values)), float(edges[-1])]
    labels = [f"{ticks[0]:.2f}", f"med {ticks[1]:.2f}", f"{ticks[2]:.2f}"]
    for t, lab in zip(ticks, labels):
        # map value to x
        tx = x0 + (t - edges[0]) / (edges[-1] - edges[0] + 1e-12) * plot_w
        draw.line([(tx, y0), (tx, y1)], fill=(230, 230, 230))
        tw = draw.textlength(lab)
        draw.text((tx - tw / 2, y1 + 6), lab, fill=(0, 0, 0))

    # Y ticks (0, max)
    for yy, lab in [(y1, "0"), (y0, str(max_count))]:
        draw.line([(x0, yy), (x1, yy)], fill=(0, 0, 0))
        draw.text((x0 - 40, yy - 8), lab, fill=(0, 0, 0))

    # Titles
    title = "Score Distribution (per-clip reconstruction MSE over masked patches)"
    tw = draw.textlength(title)
    draw.text((width / 2 - tw / 2, 12), title, fill=(0, 0, 0))

    xlabel = "score"
    tw = draw.textlength(xlabel)
    draw.text((x0 + plot_w / 2 - tw / 2, height - margin_b + 40), xlabel, fill=(0, 0, 0))

    ylabel = "count"
    draw.text((10, y0 + plot_h / 2 - 6), ylabel, fill=(0, 0, 0))

    img.save(out_path)

    stats = _format_stats(values)
    # Also save stats JSON next to the image
    stats_path = os.path.splitext(out_path)[0] + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Plot distribution of per-clip scores from JSONL")
    ap.add_argument("scores", nargs="?", default="scores.jsonl", help="Path to scores JSONL")
    ap.add_argument("--out", default="score_distribution.png", help="Output image path")
    ap.add_argument("--bins", type=int, default=40, help="Number of histogram bins")
    args = ap.parse_args()

    stats = plot_distribution(args.scores, out_path=args.out, bins=args.bins)
    print(json.dumps({"out": args.out, "stats": stats}))

