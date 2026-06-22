"""Corrected domain-confound clincher: does the TRAINED predictor's novelty score
separate Waymo/BDD from nuScenes? (The masked-gap/untrained novelty does not, which
is consistent with it scoring ~chance on the cross-dataset benchmark; the trained
predictor is the one that produced AP 0.89, so it is the score to test.)

Loads a trained predictor checkpoint, computes per-clip novelty
  = 1 - cos(predictor(masked_emb), clean_emb)
for a sample of nuScenes clips + the 22 Waymo/BDD clips, and reports the AUC for
'is this a foreign (Waymo/BDD) clip?'. High AUC => the trained predictor's novelty
is largely a domain detector.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CKPT = "/gpfs/u/scratch/EFCM/EFCMdvlr/jepa/experiments/main/2094741/runs/samples_16__seed1/training/checkpoints/best_model.pt"
EXPA = "/gpfs/u/scratch/EFCM/EFCMdvlr/manifests/expA_bench.jsonl"
POOL = "/gpfs/u/scratch/EFCM/EFCMdvlr/label_pool/label_pool.jsonl"


def main():
    import torch
    from jepa.config import load_and_resolve_config
    from jepa.data import MaskTubelet, VideoProcessor
    from jepa.pipeline import _build_model, _load_model_weights, resolve_device
    from PIL import Image
    from sklearn.metrics import roc_auc_score

    cfg = load_and_resolve_config("configs/benchmark_cluster.yaml")
    device = resolve_device("auto")
    model = _build_model(cfg).to(device).eval()
    ckpt = torch.load(CKPT, map_location=device)
    _load_model_weights(model, ckpt)
    model.eval()
    tx = MaskTubelet(mask_ratio=0.5, patch_size=16, seed=42, frame_processor=VideoProcessor(size=224))

    def trained_novelty(fp):
        ims = [Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC) for p in fp]
        nov = []
        for s in range(0, len(ims) - 1, 2):
            out = tx(ims[s:s+2])
            clean = out["clean_frames"].unsqueeze(0).to(device)
            masked = out["masked_frames"].unsqueeze(0).to(device)
            mf = out["mask_frac"].reshape(1, 1).to(device)
            ce, pe = model(clean, masked, mf)
            ce = ce.float().cpu().numpy()[0]; pe = pe.float().cpu().numpy()[0]
            nov.append(1 - np.dot(ce, pe)/(np.linalg.norm(ce)*np.linalg.norm(pe)+1e-12))
        return float(np.mean(nov))

    # foreign clips (all 22)
    foreign = []
    with torch.inference_mode():
        for l in open(EXPA):
            r = json.loads(l)
            if r["clip_id"].startswith(("waymo", "bdd")):
                foreign.append(trained_novelty(r["frame_paths"]))
        # nuScenes sample (first ~80 of label pool, absolute paths)
        nusc = []
        for i, l in enumerate(open(POOL)):
            if i >= 80:
                break
            r = json.loads(l)
            nusc.append(trained_novelty(r["frame_paths"]))
    foreign, nusc = np.array(foreign), np.array(nusc)
    y = np.concatenate([np.zeros(len(nusc)), np.ones(len(foreign))])
    score = np.concatenate([nusc, foreign])
    print("\n=== TRAINED-PREDICTOR novelty by domain ===")
    print(f"  nuScenes (n={len(nusc)}): mean novelty = {nusc.mean():.3f}")
    print(f"  Waymo/BDD (n={len(foreign)}): mean novelty = {foreign.mean():.3f}")
    print(f"  AUC for 'is this Waymo/BDD?' from trained-predictor novelty = {roc_auc_score(y, score):.3f}")
    print("  (High AUC => the trained predictor's novelty is a domain detector,")
    print("   which is what inflated the cross-dataset benchmark to AP 0.89.)")


if __name__ == "__main__":
    main()
