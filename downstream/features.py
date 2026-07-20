# ============================================================================
# Feature extraction utilities for downstream anomaly prediction.
# - Frozen encoder / quantizer are used as feature backbones.
# - Variable-wise max pooling aggregates multi-variable inputs.
# ============================================================================

import torch

# ---------------------------------------------------------------------------
# Global handles (registered once from eval / downstream entry)
# ---------------------------------------------------------------------------
encoder_online = None
quantizer_online = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_feature_models(_encoder_online, _quantizer_online, _device=None):
    """
    Register frozen backbone models for feature extraction.
    """
    global encoder_online, quantizer_online, device
    encoder_online = _encoder_online
    quantizer_online = _quantizer_online
    if _device is not None:
        device = _device


def instance_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Instance normalization over (P, L) for each variable.
    """
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True) + 1e-6
    return (x - mean) / std


def extract_feature_maxpoolV(
    x_patch: torch.Tensor,
    *,
    use_probs: bool = True,
    feature_projector=None,
):
    """
    Variable-wise max pooled feature extraction.

    Input:
        x_patch: (B, P, L, V)

    Output:
        feat: (B, P, K) if use_probs=True
              (B, P, D) if use_probs=False
    """
    assert x_patch.ndim == 4, f"Expected (B,P,L,V), got {x_patch.shape}"
    B, P, L, V = x_patch.shape

    # ------------------------------------------------------------
    # (B,P,L,V) -> (B,V,P,L,1) -> (B*V,P,L,1)
    # ------------------------------------------------------------
    x = (
        x_patch
        .permute(0, 3, 1, 2)        # (B,V,P,L)
        .unsqueeze(-1)              # (B,V,P,L,1)
        .contiguous()
        .view(B * V, P, L, 1)       # (B*V,P,L,1)
        .to(device)
    )

    x = instance_normalize(x)

    # ------------------------------------------------------------
    # Encoder
    # encoder may return tensor or tuple → always take first
    # ------------------------------------------------------------
    h = encoder_online(x)
    if isinstance(h, tuple):
        h = h[0]
    # h: (B*V, P, D)

    # ------------------------------------------------------------
    # Quantizer 
    # ------------------------------------------------------------
    q = quantizer_online(h)
    if not isinstance(q, tuple):
        raise RuntimeError("Quantizer must return a tuple")

    if len(q) == 2:
        probs, z_q = q
    elif len(q) == 3:
        probs, z_q, _ = q
    else:
        raise RuntimeError(f"Unexpected quantizer output length: {len(q)}")

    # Select feature type
    feat_v = probs if use_probs else z_q
    # feat_v: (B*V, P, K) or (B*V, P, D)

    last_dim = feat_v.shape[-1]

    # ------------------------------------------------------------
    # Variable-wise max pooling
    # (B*V,P,*) -> (B,P,V,*) -> (B,P,*)
    # ------------------------------------------------------------
    feat_v = (
        feat_v
        .view(B, V, P, last_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    feat = feat_v.max(dim=2).values   # (B,P,last_dim)

    if feature_projector is not None:
        feat = feature_projector(feat)

    return feat
