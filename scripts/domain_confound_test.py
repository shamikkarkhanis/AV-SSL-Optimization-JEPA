"""Clincher for the domain-confound section: show that JEPA novelty / embeddings
detect *which dataset* a clip is from, independent of interestingness.

Test A: does the novelty score alone separate Waymo/BDD from nuScenes? (AUC)
Test B: can a classifier predict dataset-of-origin from the frozen embedding? (AUC)
High values => the 'novelty' signal is largely a domain detector, which is what
inflated the cross-dataset benchmark.
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CACHE = "/gpfs/u/scratch/EFCM/EFCMdvlr/supervised_emb.npz"
EXPA = "/gpfs/u/scratch/EFCM/EFCMdvlr/manifests/expA_bench.jsonl"


def main():
    import torch
    from jepa.config import load_and_resolve_config
    from jepa.data import MaskTubelet, VideoProcessor
    from jepa.pipeline import _build_model, resolve_device
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score

    cache = np.load(CACHE, allow_pickle=True)
    nusc_emb, nusc_nov = cache["emb"], cache["nov"]  # all nuScenes
    print(f"nuScenes (cached): {len(nusc_nov)} clips")

    cfg = load_and_resolve_config("configs/benchmark_cluster.yaml")
    device = resolve_device("auto")
    model = _build_model(cfg).to(device).eval()
    tx = MaskTubelet(mask_ratio=0.5, patch_size=16, seed=42, frame_processor=VideoProcessor(size=224))

    def embed(fp):
        ims = [Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC) for p in fp]
        ce_l, nv_l = [], []
        for s in range(0, len(ims) - 1, 2):
            out = tx(ims[s:s+2])
            clean = out["clean_frames"].unsqueeze(0).to(device)
            masked = out["masked_frames"].unsqueeze(0).to(device)
            ce = model.encoder.get_embedding(clean).float().cpu().numpy()[0]
            me = model.encoder.get_embedding(masked).float().cpu().numpy()[0]
            ce_l.append(ce); nv_l.append(1 - np.dot(ce, me)/(np.linalg.norm(ce)*np.linalg.norm(me)+1e-12))
        return np.mean(ce_l, axis=0), float(np.mean(nv_l))

    f_emb, f_nov = [], []
    with torch.inference_mode():
        for l in open(EXPA):
            r = json.loads(l)
            if not r["clip_id"].startswith(("waymo", "bdd")):
                continue
            e, n = embed(r["frame_paths"]); f_emb.append(e); f_nov.append(n)
    f_emb, f_nov = np.array(f_emb), np.array(f_nov)
    print(f"foreign (Waymo/BDD): {len(f_nov)} clips")

    y = np.concatenate([np.zeros(len(nusc_nov)), np.ones(len(f_nov))])  # 1 = foreign
    nov = np.concatenate([nusc_nov, f_nov])
    print("\n=== DOMAIN-CONFOUND EVIDENCE ===")
    print(f"mean novelty:  nuScenes={nusc_nov.mean():.3f}   Waymo/BDD={f_nov.mean():.3f}")
    print(f"Test A: novelty-score AUC for 'is this Waymo/BDD?'   = {roc_auc_score(y, nov):.3f}")
    X = np.vstack([nusc_emb, f_emb])
    Xs = StandardScaler().fit_transform(X)
    auc = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                          Xs, y, cv=5, scoring="roc_auc")
    print(f"Test B: embedding->domain classifier AUC (5-fold)    = {auc.mean():.3f} ± {auc.std():.3f}")
    print("(AUC near 1.0 => domain is trivially encoded; the 'novelty' signal rides on it.)")


if __name__ == "__main__":
    main()
