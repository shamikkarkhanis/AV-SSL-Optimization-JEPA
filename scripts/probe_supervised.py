"""Early read: can a classifier on frozen V-JEPA embeddings separate
interesting vs routine clips? Scene-disjoint cross-validated probe.

Extracts one mean-pooled embedding per labeled clip, runs GroupKFold (by scene)
logistic regression, and reports out-of-fold AP / ROC-AUC vs. chance. Also
computes the unsupervised masked-gap novelty AP on the SAME clips for contrast.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from jepa.config import load_and_resolve_config
from jepa.data import TubeletDataset, MaskTubelet, VideoProcessor
from jepa.pipeline import _build_model, resolve_device
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark_cluster.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = load_and_resolve_config(args.config)
    device = resolve_device("auto")
    labels = {json.loads(l)["clip_id"]: json.loads(l)["binary_label"]
              for l in open(args.labels) if json.loads(l).get("binary_label") is not None}

    ds = TubeletDataset(
        manifest_path=args.manifest, data_root=None, tubelet_size=2,
        transform=MaskTubelet(mask_ratio=0.5, patch_size=16, seed=42,
                              frame_processor=VideoProcessor(size=224)),
        split="label", frames_per_clip=16,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=8)
    model = _build_model(cfg).to(device).eval()

    clean_by_clip = defaultdict(list)
    novelty_by_clip = defaultdict(list)
    scene_by_clip = {}
    with torch.inference_mode():
        for batch in loader:
            cid = str(batch["meta"]["clip_id"][0])
            if cid not in labels:
                continue
            clean = batch["clean_frames"].to(device)
            masked = batch["masked_frames"].to(device)
            ce = model.encoder.get_embedding(clean).float().cpu().numpy()[0]
            me = model.encoder.get_embedding(masked).float().cpu().numpy()[0]
            clean_by_clip[cid].append(ce)
            cos = np.dot(ce, me) / (np.linalg.norm(ce) * np.linalg.norm(me) + 1e-12)
            novelty_by_clip[cid].append(1.0 - cos)
            scene_by_clip[cid] = str(batch["meta"]["scene_id"][0])

    clips = [c for c in clean_by_clip if c in labels]
    X = np.stack([np.mean(clean_by_clip[c], axis=0) for c in clips])
    y = np.array([int(labels[c]) for c in clips])
    groups = np.array([scene_by_clip[c] for c in clips])
    novelty = np.array([float(np.mean(novelty_by_clip[c])) for c in clips])

    n_scenes = len(set(groups))
    folds = min(args.folds, n_scenes)
    gkf = GroupKFold(n_splits=folds)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    chance = y.mean()
    print(f"\n=== SUPERVISED PROBE ({len(y)} clips, {n_scenes} scenes, {y.sum()} pos) ===")
    print(f"chance AP (positive rate):     {chance:.3f}")
    print(f"supervised AP (scene-CV):      {average_precision_score(y, oof):.3f}")
    print(f"supervised ROC-AUC (scene-CV): {roc_auc_score(y, oof):.3f}")
    print(f"--- contrast on same clips ---")
    print(f"unsupervised novelty AP:       {average_precision_score(y, novelty):.3f}")
    print(f"unsupervised novelty ROC-AUC:  {roc_auc_score(y, novelty):.3f}")


if __name__ == "__main__":
    main()
