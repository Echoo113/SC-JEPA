# engine/losses.py
import torch
import torch.nn.functional as F


def kl_loss_fine(logits_pred, p_future, temperature):
    """
    Patch-level KL divergence between predicted and target code distributions.
    """
    logp_pred = F.log_softmax(logits_pred / temperature, dim=-1)
    return F.kl_div(logp_pred, p_future, reduction="batchmean")


def kl_loss_coarse(logits_pred_coarse, p_future_coarse, temperature):
    """
    Coarse-scale KL divergence (single patch).
    """
    logp_pred = F.log_softmax(logits_pred_coarse / temperature, dim=-1)
    return F.kl_div(logp_pred, p_future_coarse, reduction="batchmean")


def mse_alignment_loss(z_pred, z_q_future):
    """
    Latent-space alignment loss between predicted and target quantized latents.
    """
    return F.mse_loss(z_pred, z_q_future.detach())


def vq_losses(h_past, z_q_past):
    """
    Soft codebook quantization and commitment losses.
    """
    loss_q = F.mse_loss(z_q_past, h_past.detach())
    loss_commit = F.mse_loss(h_past, z_q_past.detach())
    return loss_q, loss_commit


def entropy_losses(p_past, eps=1e-6):
    """
    Entropy-based regularization terms:
    - sample-wise entropy (minimized)
    - batch-wise entropy (maximized via negative sign)
    """
    probs_safe = torch.clamp(p_past, min=eps, max=1 - eps)

    entropy_sample = -(probs_safe * torch.log(probs_safe)).sum(dim=-1).mean()

    avg_probs = probs_safe.mean(dim=(0, 1))
    batch_entropy = -(avg_probs * torch.log(avg_probs)).sum()
    entropy_batch = -batch_entropy

    return entropy_sample, entropy_batch
