# ============================================================================
# Utility Functions
# ============================================================================
# Core utilities for instance-wise normalization, EMA updates,
# and coarse-scale patch construction.
#
# Design assumptions:
# - Normalization is performed per-sample and per-channel,
#   aggregating over patch and temporal dimensions.
# - Stored mean/std are instance statistics and must only be
#   reused for tensors derived from the same instance and scale.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def instance_normalize(batch: torch.Tensor):
    """
    Instance-wise normalization for patch-based time series.

    Args:
        batch: Tensor of shape (B, P, L, C),
               where normalization is applied independently
               for each sample and each channel.

    Returns:
        normalized: Tensor of shape (B, P, L, C)
        mean: Instance mean, shape (B, 1, 1, C)
        std:  Instance std,  shape (B, 1, 1, C)
    """
    mean = batch.mean(dim=(1, 2), keepdim=True)
    std = batch.std(dim=(1, 2), keepdim=True) + 1e-6
    normalized = (batch - mean) / std
    return normalized, mean, std


def reverse_instance_normalize(normed: torch.Tensor,
                               mean: torch.Tensor,
                               std: torch.Tensor):
    """
    Reverse instance-wise normalization using stored instance statistics.

    This function assumes the normalized tensor has layout (B, N, C, L),
    and the provided mean/std originate from the corresponding
    instance_normalize call on the same sample.

    Args:
        normed: Tensor of shape (B, N, C, L)
        mean:   Tensor of shape (B, 1, 1, C)
        std:    Tensor of shape (B, 1, 1, C)

    Returns:
        Tensor of shape (B, N, C, L)
    """
    mean = mean.permute(0, 1, 3, 2)
    std = std.permute(0, 1, 3, 2)
    return normed * std + mean


@torch.no_grad()
def update_ema(model_online: nn.Module,
               model_ema: nn.Module,
               decay: float):
    """
    Exponential moving average (EMA) update.

    model_ema ← decay * model_ema + (1 - decay) * model_online
    """
    for p_online, p_ema in zip(model_online.parameters(),
                               model_ema.parameters()):
        p_ema.data.mul_(decay).add_(p_online.data, alpha=1 - decay)


def coarse_scale_to_patch(x_input: torch.Tensor,
                          num_patches: int = 5,
                          patch_len: int = 20,
                          add_patch_dim: bool = False):
    """
    Construct a coarse-scale patch by averaging consecutive fine-scale patches.

    Args:
        x_input: Tensor of shape (B, P, L, C)
        num_patches: Number of fine-scale patches (P)
        patch_len: Length of each fine-scale patch (L)
        add_patch_dim: If True, output has an explicit patch dimension

    Returns:
        Tensor of shape:
        - (B, L, C) if add_patch_dim is False
        - (B, 1, L, C) if add_patch_dim is True
    """
    B, P, L, C = x_input.shape

    x_flat = x_input.view(B, P * L, C)
    x_grouped = x_flat.view(B, L, P, C)
    new_patch = x_grouped.mean(dim=2)

    if add_patch_dim:
        new_patch = new_patch.unsqueeze(1)

    return new_patch
