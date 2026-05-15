from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def mask_kl_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL loss for probability masks that sum to one across stems."""
    pred = torch.clamp(pred_masks, min=eps, max=1.0)
    target = torch.clamp(target_masks, min=eps, max=1.0)
    return F.kl_div(pred.log(), target, reduction="batchmean")


def spectral_convergence_loss(pred_mag: torch.Tensor, target_mag: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred_mag.flatten(start_dim=1)
    target = target_mag.flatten(start_dim=1)
    diff = torch.linalg.norm(target - pred, dim=1)
    denom = torch.linalg.norm(target, dim=1) + eps
    return (diff / denom).mean()


def multi_scale_mag_loss(
    pred_mag: torch.Tensor,
    target_mag: torch.Tensor,
    *,
    scales: Iterable[int] = (1, 2, 4),
) -> torch.Tensor:
    """Multi-resolution magnitude loss via average pooling over frequency-time maps."""
    losses = []
    for scale in scales:
        if scale <= 1:
            pooled_pred = pred_mag
            pooled_target = target_mag
        else:
            pooled_pred = F.avg_pool2d(pred_mag, kernel_size=scale, stride=scale, ceil_mode=True)
            pooled_target = F.avg_pool2d(target_mag, kernel_size=scale, stride=scale, ceil_mode=True)
        l1 = F.l1_loss(pooled_pred, pooled_target)
        sc = spectral_convergence_loss(pooled_pred, pooled_target)
        losses.append(l1 + 0.5 * sc)
    return torch.stack(losses).mean()


def separation_consistency_loss(pred_masks: torch.Tensor) -> torch.Tensor:
    """Encourage masks to sum to one across the source dimension."""
    total = pred_masks.sum(dim=1, keepdim=True)
    target = torch.ones_like(total)
    return F.mse_loss(total, target)


def sisdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio loss.
    Operates on (batch, stems, freq, time) or (batch, freq, time).
    """
    # Flatten to (..., samples)
    est = est.flatten(start_dim=-2)
    ref = ref.flatten(start_dim=-2)
    
    # Calculate scale factor alpha
    dot = torch.sum(est * ref, dim=-1, keepdim=True)
    ref_pow = torch.sum(ref**2, dim=-1, keepdim=True) + eps
    alpha = dot / ref_pow
    
    # Calculate SI-SDR
    target = alpha * ref
    noise = est - target
    
    target_pow = torch.sum(target**2, dim=-1, keepdim=True) + eps
    noise_pow = torch.sum(noise**2, dim=-1, keepdim=True) + eps
    
    # We use negative SI-SDR as loss
    sisdr = 10 * torch.log10(target_pow / noise_pow + eps)
    return -torch.mean(sisdr)


def composite_separation_loss(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
    est_mags: torch.Tensor,
    target_mags: torch.Tensor,
    *,
    mask_weight: float = 0.35,
    mag_weight: float = 0.35,
    consistency_weight: float = 0.05,
    kl_weight: float = 0.05,
    sisdr_weight: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted multi-objective loss for source separation training/eval."""
    mask_loss = F.smooth_l1_loss(pred_masks, target_masks)
    mag_loss = multi_scale_mag_loss(torch.log1p(est_mags), torch.log1p(target_mags))
    consistency_loss = separation_consistency_loss(pred_masks)
    kl_loss = mask_kl_loss(pred_masks, target_masks)
    sisdr_l = sisdr_loss(est_mags, target_mags)

    total = (
        mask_weight * mask_loss
        + mag_weight * mag_loss
        + consistency_weight * consistency_loss
        + kl_weight * kl_loss
        + sisdr_weight * sisdr_l
    )
    components = {
        "mask_loss": mask_loss,
        "mag_loss": mag_loss,
        "consistency_loss": consistency_loss,
        "kl_loss": kl_loss,
        "sisdr_loss": sisdr_l,
    }
    return total, components


