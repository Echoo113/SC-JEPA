# ============================================================================
# Quantizer Module
# ============================================================================
# Soft codebook quantizer using cosine similarity
# Used in JEPA + soft codebook framework
#
# Input : (B, num_patches, latent_dim)
# Output:
#   - probs: (B, num_patches, num_codes)   soft assignment
#   - z_q  : (B, num_patches, latent_dim)  quantized latent
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    """
    Soft codebook quantizer with cosine similarity.

    Uses soft assignments instead of hard argmin to improve stability.
    Codebook vectors and latent vectors are L2-normalized before similarity.
    """
    def __init__(
        self,
        num_codes: int,
        embedding_dim: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_codes, 1.0 / num_codes
        )

    def forward(self, z: torch.Tensor):
        """
        z: (B, N, D) latent vectors from encoder

        Returns:
            probs: (B, N, K) soft code assignment
            z_q  : (B, N, D) quantized latent vectors
        """
        # Normalize latent vectors
        z_norm = F.normalize(z, dim=-1)  # (B, N, D)

        # Normalize codebook
        codebook = F.normalize(self.embedding.weight, dim=-1)  # (K, D)

        # Cosine similarity
        sim = torch.einsum("bnd,kd->bnk", z_norm, codebook)  # (B, N, K)

        # Convert similarity to distance-like score
        dists = 1.0 - sim

        # Soft assignment
        probs = F.softmax(-dists / self.temperature, dim=-1)  # (B, N, K)

        # Quantized latent
        z_q = torch.einsum("bnk,kd->bnd", probs, codebook)  # (B, N, D)

        return probs, z_q
