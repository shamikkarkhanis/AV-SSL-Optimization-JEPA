"""Paper-grade supervised triage experiment on frozen V-JEPA embeddings.

Combines all labeled nuScenes clips (new label_pool + existing extra labels) into
one training pool, holds out the 83-clip benchmark, and reports:
  - Repeated scene-disjoint CV (mean +/- std) for: supervised head, unsupervised
    JEPA novelty, kNN-density novelty, vs chance.
  - A held-out test on the 83-clip benchmark (train on the full pool, scenes
    excluded; evaluate supervised + baselines).
Embeddings are extracted once on GPU and cached to npz so analysis re-runs are instant.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CLUSTER_MAN = "/gpfs/u/barn/EFCM/shared/data/manifests"
TRAINVAL01_SAMPLES = "/gpfs/u/barn/EFCM/shared/data/raw/v1.0-trainval01/samples"


def _resolve(fp):
    if os.path.isabs(fp):
        return fp if os.path.exists(fp) else None
    cand = os.path.join(TRAINVAL01_SAMPLES, fp)
    return cand if os.path.exists(cand) else None


def assemble_clips(label_pool_manifest, label_pool_labels, benchmark_labels):
    """Return dict clip_id -> {frame_paths, label, scene_id, group, source}."""
    # frame paths from every manifest
    frames = {}
    for p in glob.glob(os.path.join(CLUSTER_MAN, "*.jsonl")) + [label_pool_manifest]:
        if "evaluation_labels" in p:
            continue
        for l in open(p):
            l = l.strip()
            if not l:
                continue
            try: r = json.loads(l)
            except: continue
            if "frame_paths" in r:
                frames.setdefault(r["clip_id"], (r["frame_paths"], r.get("scene_id", "")))

    bench_ids, bench_scenes = set(), set()
    for l in open(benchmark_labels):
        r = json.loads(l)
        if r.get("binary_label") is not None:
            bench_ids.add(r["clip_id"]); bench_scenes.add(r.get("scene_id", ""))

    def resolved_or_none(fp):
        out = [_resolve(p) for p in fp]
        return out if all(out) else None

    clips = {}
    dropped = 0
    # new label-pool labels
    for l in open(label_pool_labels):
        r = json.loads(l)
        if r.get("binary_label") is None or r["clip_id"] not in frames:
            continue
        fp, sc = frames[r["clip_id"]]
        rp = resolved_or_none(fp)
        if rp is None:
            dropped += 1; continue
        clips[r["clip_id"]] = dict(frame_paths=rp, label=int(r["binary_label"]),
                                   scene_id=r.get("scene_id", sc), group="pool", source="new")
    # existing nuScenes labels from all cluster label files
    for p in glob.glob(os.path.join(CLUSTER_MAN, "*evaluation_labels.jsonl")):
        for l in open(p):
            r = json.loads(l)
            cid = r.get("clip_id", "")
            if r.get("binary_label") is None or cid.startswith(("waymo", "bdd")):
                continue
            if cid in clips or cid not in frames:
                continue
            fp, sc = frames[cid]
            rp = resolved_or_none(fp)
            if rp is None:
                dropped += 1; continue
            grp = "benchmark" if cid in bench_ids else "pool"
            clips[cid] = dict(frame_paths=rp, label=int(r["binary_label"]),
                              scene_id=r.get("scene_id", sc), group=grp, source="existing")
    if dropped:
        print(f"  (dropped {dropped} clips with missing frames)")
    return clips, bench_scenes


def extract_embeddings(clips, config_path, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    import torch
    from jepa.config import load_and_resolve_config
    from jepa.data import TubeletDataset, MaskTubelet, VideoProcessor
    from jepa.pipeline import _build_model, resolve_device
    cfg = load_and_resolve_config(config_path)
    device = resolve_device("auto")
    model = _build_model(cfg).to(device).eval()
    tx = MaskTubelet(mask_ratio=0.5, patch_size=16, seed=42, frame_processor=VideoProcessor(size=224))

    ids = list(clips.keys())
    emb = np.zeros((len(ids), 1024), dtype=np.float32)
    nov = np.zeros(len(ids), dtype=np.float32)
    import torch as T
    from PIL import Image
    def load_clip(fp):
        # Paths are pre-resolved to absolute in assemble_clips. Resize to 224x224
        # like JEPADataset before masking (patch_size=16 must divide H,W).
        ims = [Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC) for p in fp]
        return ims
    with T.inference_mode():
        for i, cid in enumerate(ids):
            ims = load_clip(clips[cid]["frame_paths"])
            ce_list, nov_list = [], []
            # process in 2-frame tubelets like the pipeline
            for s in range(0, len(ims) - 1, 2):
                tub = ims[s:s+2]
                out = tx(tub)
                clean = out["clean_frames"].unsqueeze(0).to(device)
                masked = out["masked_frames"].unsqueeze(0).to(device)
                ce = model.encoder.get_embedding(clean).float().cpu().numpy()[0]
                me = model.encoder.get_embedding(masked).float().cpu().numpy()[0]
                ce_list.append(ce)
                nov_list.append(1.0 - np.dot(ce, me)/(np.linalg.norm(ce)*np.linalg.norm(me)+1e-12))
            emb[i] = np.mean(ce_list, axis=0)
            nov[i] = float(np.mean(nov_list))
            if (i+1) % 50 == 0:
                print(f"  embedded {i+1}/{len(ids)}", flush=True)
    out = dict(
        ids=np.array(ids), emb=emb, nov=nov,
        label=np.array([clips[c]["label"] for c in ids]),
        scene=np.array([clips[c]["scene_id"] for c in ids]),
        group=np.array([clips[c]["group"] for c in ids]),
    )
    np.savez(cache, **out)
    return out


def analyze(data, repeats=10, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import average_precision_score, roc_auc_score
    from jepa.baselines import embedding_density_novelty

    g = data["group"]
    pool = g == "pool"
    bench = g == "benchmark"
    Xp, yp, sp, novp = data["emb"][pool], data["label"][pool], data["scene"][pool], data["nov"][pool]
    print(f"\nTRAINING POOL: {len(yp)} clips, {len(set(sp))} scenes, {yp.sum()} pos ({yp.mean():.2f})")

    # Repeated scene-disjoint CV for the supervised head
    aps, aucs = [], []
    for rep in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=rep)
        oof = np.zeros(len(yp))
        for tr, te in sgkf.split(Xp, yp, sp):
            sc = StandardScaler().fit(Xp[tr])
            clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
            clf.fit(sc.transform(Xp[tr]), yp[tr])
            oof[te] = clf.predict_proba(sc.transform(Xp[te]))[:, 1]
        aps.append(average_precision_score(yp, oof)); aucs.append(roc_auc_score(yp, oof))
    dens = embedding_density_novelty(Xp, k=5)
    chance = yp.mean()

    def line(name, ap, auc): return f"  {name:<34} AP={ap:.3f}  ROC-AUC={auc:.3f}"
    print("\n=== TRAINING-POOL (scene-disjoint CV, mean±std over %d repeats) ===" % repeats)
    print(f"  chance                             AP={chance:.3f}")
    print(line("unsupervised JEPA novelty", average_precision_score(yp, novp), roc_auc_score(yp, novp)))
    print(line("kNN-density novelty", average_precision_score(yp, dens), roc_auc_score(yp, dens)))
    print(f"  supervised head                    AP={np.mean(aps):.3f}±{np.std(aps):.3f}  "
          f"ROC-AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")

    # Held-out benchmark test (train on pool clips whose scenes are NOT in benchmark)
    if bench.sum() > 0:
        Xb, yb, novb = data["emb"][bench], data["label"][bench], data["nov"][bench]
        bench_scenes = set(data["scene"][bench])
        keep = np.array([s not in bench_scenes for s in sp])
        sc = StandardScaler().fit(Xp[keep])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        clf.fit(sc.transform(Xp[keep]), yp[keep])
        pred = clf.predict_proba(sc.transform(Xb))[:, 1]
        densb = embedding_density_novelty(Xb, k=5)
        print(f"\n=== HELD-OUT BENCHMARK ({len(yb)} clips, {yb.sum()} pos, "
              f"trained on {keep.sum()} scene-disjoint pool clips) ===")
        print(f"  chance                             AP={yb.mean():.3f}")
        print(line("unsupervised JEPA novelty", average_precision_score(yb, novb), roc_auc_score(yb, novb)))
        print(line("kNN-density novelty", average_precision_score(yb, densb), roc_auc_score(yb, densb)))
        print(line("supervised head", average_precision_score(yb, pred), roc_auc_score(yb, pred)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark_cluster.yaml")
    ap.add_argument("--label-pool-manifest", required=True)
    ap.add_argument("--label-pool-labels", required=True)
    ap.add_argument("--benchmark-labels", required=True)
    ap.add_argument("--cache", default="/gpfs/u/scratch/EFCM/EFCMdvlr/supervised_emb.npz")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    clips, _ = assemble_clips(args.label_pool_manifest, args.label_pool_labels, args.benchmark_labels)
    from collections import Counter
    print("assembled clips:", len(clips), "| group:",
          dict(Counter(c["group"] for c in clips.values())),
          "| source:", dict(Counter(c["source"] for c in clips.values())))
    data = extract_embeddings(clips, args.config, args.cache)
    analyze(data, repeats=args.repeats)


if __name__ == "__main__":
    main()
