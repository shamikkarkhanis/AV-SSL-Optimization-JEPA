import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image


@torch.no_grad()
def _to_uint8_img(chw: torch.Tensor) -> Image.Image:
    """
    Convert tensor in CHW with values in [0,1] (or close) to a PIL Image.
    """
    chw = chw.clamp(0.0, 1.0)
    arr = (chw * 255.0).round().byte().cpu()
    if arr.dim() != 3 or arr.shape[0] not in (1, 3):
        # Fallback: try to squeeze and repeat to 3 channels
        if arr.dim() == 2:
            arr = arr.unsqueeze(0)
        if arr.shape[0] == 1:
            arr = arr.repeat(3, 1, 1)
        elif arr.shape[0] > 3:
            arr = arr[:3]
    # CHW -> HWC
    arr = arr.permute(1, 2, 0).contiguous().numpy()
    mode = "L" if arr.shape[2] == 1 else "RGB"
    return Image.fromarray(arr, mode=mode)


@torch.no_grad()
def _unnormalize_rgb(x_btcwh: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize using ImageNet mean/std if 3 channels, else return as-is.
    Expects shape (B, T, C, H, W).
    """
    B, T, C, H, W = x_btcwh.shape
    if C == 3:
        device = x_btcwh.device
        dtype = x_btcwh.dtype
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)[None, None, :, None, None]
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)[None, None, :, None, None]
        return x_btcwh * std + mean
    return x_btcwh


@torch.no_grad()
def _reconstruct_from_logits(
    out_logits: torch.Tensor,
    mask: torch.Tensor,
    pv_btcwh: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """
    Reconstruct a full video from masked-patch predictions and mask.
    Returns shape (B, T, C, H, W) with zeros in unmasked regions.
    """
    B, T, C, H, W = pv_btcwh.shape
    ps = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
    ts = getattr(cfg, "tubelet_size", 1)

    t_tokens = T // ts
    h_tokens = H // ps
    w_tokens = W // ps
    total_tokens = t_tokens * h_tokens * w_tokens
    feat_dim = C * ts * ps * ps

    preds = out_logits  # (B, N_masked, feat_dim)
    if preds.dim() != 3 or preds.shape[0] != B or preds.shape[2] != feat_dim:
        raise ValueError("Unexpected logits shape for reconstruction")

    full_tokens = torch.zeros((B, total_tokens, feat_dim), device=preds.device, dtype=preds.dtype)
    full_tokens[mask] = preds.reshape(-1, feat_dim)

    x = full_tokens.view(B, t_tokens, h_tokens, w_tokens, ts, ps, ps, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, C)
    x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)
    return x


@torch.no_grad()
def _patchify_and_stats(pv_btcwh: torch.Tensor, cfg):
    """
    Convert pv_btcwh (B,T,C,H,W) into tokens (B, total_tokens, F) and compute per-token
    mean/std over feature dimension using unnormalized GT (in [0,1]) to mirror MAE targets.
    Returns: videos_patch (B, N, F), mean (B, N, 1), std (B, N, 1)
    """
    # Unnormalize GT first for stats as in compute_per_sample_loss
    frames = _unnormalize_rgb(pv_btcwh)

    B, T, C, H, W = frames.shape
    ts = getattr(cfg, "tubelet_size", 1)
    ps = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]

    frames = frames.view(
        B,
        T // ts,
        ts,
        C,
        H // ps,
        ps,
        W // ps,
        ps,
    )
    # (B, T//ts, H//ps, W//ps, ts, ps, ps, C)
    frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()

    # Flatten patch dims to feature dimension
    videos_patch = frames.view(
        B,
        (T // ts) * (H // ps) * (W // ps),
        ts * ps * ps * C,
    )
    # Per-token mean/std over feature dim
    mean = videos_patch.mean(dim=-1, keepdim=True)
    var = videos_patch.var(dim=-1, unbiased=True, keepdim=True)
    std = (var.clamp_min(0.0).sqrt() + 1e-6)
    return videos_patch, mean, std


@torch.no_grad()
def _reconstruct_from_logits_denorm(
    out_logits: torch.Tensor,
    mask: torch.Tensor,
    pv_btcwh: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """
    Denormalize predicted masked patch logits using GT per-patch mean/std, then reconstruct.
    Returns (B, T, C, H, W) in [0,1] range (clipped), masked regions filled, others zero.
    """
    B, T, C, H, W = pv_btcwh.shape
    ps = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
    ts = getattr(cfg, "tubelet_size", 1)
    t_tokens = T // ts
    h_tokens = H // ps
    w_tokens = W // ps
    total_tokens = t_tokens * h_tokens * w_tokens
    feat_dim = C * ts * ps * ps

    preds = out_logits  # (B, N_masked, F)
    if preds.dim() != 3 or preds.shape[0] != B or preds.shape[2] != feat_dim:
        raise ValueError("Unexpected logits shape for reconstruction (denorm)")

    # Per-token stats from GT
    _, mean, std = _patchify_and_stats(pv_btcwh, cfg)
    mean_masked = mean[mask].reshape(B, -1, 1)
    std_masked = std[mask].reshape(B, -1, 1)

    preds_denorm = preds * std_masked + mean_masked  # (B, N_masked, F)

    # Scatter to full token grid and reshape to video
    full_tokens = torch.zeros((B, total_tokens, feat_dim), device=preds.device, dtype=preds.dtype)
    full_tokens[mask] = preds_denorm.reshape(-1, feat_dim)

    x = full_tokens.view(B, t_tokens, h_tokens, w_tokens, ts, ps, ps, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, C)
    x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)
    # Clip to [0,1]
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def render_batch(
    step: int,
    pv_btcwh: torch.Tensor,
    *,
    pred_clip_btcwh: Optional[torch.Tensor] = None,
    out_logits: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    cfg=None,
    out_dir: str = "renders",
    frame_index: Optional[int] = None,
    side_by_side: bool = True,
    overlay_mask: bool = True,
    denorm_preds: bool = True,
) -> None:
    """
    Save GT and predicted frames for each sample in the batch.

    - pv_btcwh: (B, T, C, H, W) input frames (normalized)
    - pred_clip_btcwh: (B, T, C, H, W) predicted clip (optional)
    - out_logits: model logits over masked patches (B, N_masked, F), used if pred_clip_btcwh is None
    - mask: boolean mask over tokens (B, total_tokens)
    - cfg: model/config with patch_size/tubelet_size for reconstruction
    - frame_index: specific t to render; defaults to middle frame
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Ensure CPU for PIL conversion
    pv_btcwh = pv_btcwh.detach()
    if pred_clip_btcwh is not None:
        pred_clip_btcwh = pred_clip_btcwh.detach()

    B, T, C, H, W = pv_btcwh.shape
    t_idx = frame_index if frame_index is not None else (T // 2)
    t_idx = int(max(0, min(T - 1, t_idx)))

    # Prepare predicted clip
    if pred_clip_btcwh is None:
        if out_logits is None or mask is None or cfg is None:
            raise ValueError("Need either pred_clip_btcwh or (out_logits, mask, cfg) to render predictions")
        if denorm_preds:
            pred_clip_btcwh = _reconstruct_from_logits_denorm(out_logits, mask, pv_btcwh, cfg)
        else:
            pred_clip_btcwh = _reconstruct_from_logits(out_logits, mask, pv_btcwh, cfg)

    # Unnormalize GT for visualization
    gt_vis = _unnormalize_rgb(pv_btcwh)
    pred_vis = pred_clip_btcwh

    for b in range(B):
        gt = gt_vis[b, t_idx]  # (C, H, W)
        pr = pred_vis[b, t_idx]  # (C, H, W)

        gt_img = _to_uint8_img(gt)
        pr_img = _to_uint8_img(pr)

        gt_path = Path(out_dir) / f"batch{step:04d}_sample{b:02d}_gt_t{t_idx:02d}.png"
        pr_path = Path(out_dir) / f"batch{step:04d}_sample{b:02d}_pred_t{t_idx:02d}.png"

        # Always save individual frames (for compatibility)
        gt_img.save(gt_path)
        pr_img.save(pr_path)

        if side_by_side:
            # Optional mask overlay panel for the shown frame
            mask_panel = None
            if overlay_mask and mask is not None and cfg is not None:
                Bm, total_tokens = mask.shape
                Cc = gt.shape[0]
                ps = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
                ts = getattr(cfg, "tubelet_size", 1)

                T = pv_btcwh.shape[1]
                H = gt.shape[1]
                W = gt.shape[2]
                t_tokens = T // ts
                h_tokens = H // ps
                w_tokens = W // ps

                # indices for this frame's tubelet
                t_t = t_idx // ts
                # mask per (t_t, h, w)
                mask_b = mask[b].view(t_tokens, h_tokens, w_tokens)
                mask_hw = mask_b[t_t].float()
                # upsample to HxW
                mask_img = torch.kron(mask_hw, torch.ones(ps, ps, dtype=mask_hw.dtype))  # (H, W)
                mask_img = mask_img[:H, :W]

                # create overlay on GT
                base = gt_img.convert("RGBA")
                overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
                # alpha proportional to mask
                alpha = (mask_img.clamp(0, 1) * 160).byte().cpu().numpy()
                red = Image.new("L", base.size, 255)
                zero = Image.new("L", base.size, 0)
                aimg = Image.fromarray(alpha, mode="L")
                overlay = Image.merge("RGBA", (red, zero, zero, aimg))
                mask_panel = Image.alpha_composite(base, overlay).convert("RGB")

            # Build side-by-side panel: GT | Pred | (GT+Mask)
            panels = [gt_img, pr_img]
            if mask_panel is not None:
                panels.append(mask_panel)
            total_w = sum(p.width for p in panels)
            max_h = max(p.height for p in panels)
            canvas = Image.new("RGB", (total_w, max_h))
            x = 0
            for p in panels:
                canvas.paste(p, (x, 0))
                x += p.width

            viz_path = Path(out_dir) / f"batch{step:04d}_sample{b:02d}_viz_t{t_idx:02d}.png"
            canvas.save(viz_path)
