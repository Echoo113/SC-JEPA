import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Global Configuration (copied verbatim for predictor dependency)
# ============================================================================
LATENT_DIM = 256


# ============================================================================
# Transformer Predictor
# ============================================================================
# Predicts future patch-level code distributions from past distributions.
# This module is the core of JEPA: it learns a mapping from past → future
# entirely in the latent code space.
#
# Architecture:
#   Input projection → Positional embedding → Transformer encoder
#   → Code distribution head + Latent prediction head
#
# Input : (B, N, K)
# Output: logits (B, N, K), latent_pred (B, N, D)

class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor that maps past patch-level code distributions
    to future patch-level code distribution logits.

    Input:
        p : Tensor of shape (B, N, K)
            Soft code distributions for past patches.

    Output:
        logits      : Tensor of shape (B, N, K)
            Predicted future code distribution logits.
        latent_pred : Tensor of shape (B, N, D)
            Predicted future latent representations for auxiliary alignment.
    """
    def __init__(self, num_codes=128, nhead=4, num_layers=2,
                 hidden_dim=128, dropout=0.1, num_patches=5, latent_dim=256):
        super().__init__()
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim

        # Project code distributions into continuous hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(num_codes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout)
        )

        # Learnable positional embeddings for patch ordering
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        # Transformer encoder for modeling temporal dependencies between patches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Optional self-attention layer for modeling inter-variable relationships
        # (currently disabled in forward pass)
        self.var_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.var_norm = nn.LayerNorm(hidden_dim)

        # Output head for predicting code distribution logits
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_codes)
        )

        # Output head for predicting latent representations
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear and normalization layers for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, p: torch.Tensor):
        """
        Forward pass of the predictor.

        Args:
            p : Tensor of shape (B, N, K)
                Past patch-level code distributions.

        Returns:
            logits      : Tensor of shape (B, N, K)
            latent_pred : Tensor of shape (B, N, D)
        """
        B, N, K = p.shape

        # Project input distributions to hidden space
        x = self.input_proj(p)  # (B, N, H)

        # Optional inter-variable attention (disabled by default)
        # attn_out, _ = self.var_attention(x, x, x)
        # x = self.var_norm(x + attn_out)

        # Add positional embeddings
        x = x + self.pos_emb[:, :N, :]  # (B, N, H)

        # Transformer encoding
        x = self.transformer(x)  # (B, N, H)

        # Predict future code distributions and latent representations
        logits = self.output_proj(x)       # (B, N, K)
        latent_pred = self.latent_proj(x)  # (B, N, D)

        return logits, latent_pred


class Predictor(TransformerPredictor):
    """
    Alias for TransformerPredictor.
    Kept for compatibility with existing training scripts.
    """
    pass


# ============================================================================
# Coarse Predictor
# ============================================================================
# Predicts a single coarse-scale future patch from multiple fine-scale patches.
#
# Input : (B, 5, K)
# Output: (B, 1, K)
#
# No pooling is used. Instead, a learnable query token attends to all
# input patches via cross-attention.

class CoarsePredictor(nn.Module):
    """
    Coarse-scale predictor that maps multiple fine-scale patch code distributions
    to a single future coarse-scale patch distribution.

    Input:
        p : Tensor of shape (B, 5, K)

    Output:
        logits      : Tensor of shape (B, 1, K)
        latent_pred : Tensor of shape (B, 1, D)
    """
    def __init__(self, num_codes=128, nhead=4, num_layers=2,
                 hidden_dim=128, dropout=0.1, num_patches=5, latent_dim=256):
        super().__init__()
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        # Project input code distributions
        self.input_proj = nn.Sequential(
            nn.Linear(num_codes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout)
        )

        # Positional embeddings for input patches
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        # Learnable query token used to aggregate information
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder for processing fine-scale patches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Cross-attention: query token attends to encoded patches
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Output head for coarse-scale code distribution
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_codes)
        )

        # Output head for coarse-scale latent representation
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear and normalization layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, p: torch.Tensor):
        """
        Forward pass of the coarse predictor.

        Args:
            p : Tensor of shape (B, 5, K)

        Returns:
            logits      : Tensor of shape (B, 1, K)
            latent_pred : Tensor of shape (B, 1, D)
        """
        B, N, K = p.shape

        # Encode fine-scale patches
        x = self.input_proj(p)                 # (B, 5, H)
        x = x + self.pos_emb[:, :N, :]         # (B, 5, H)
        x = self.transformer(x)                # (B, 5, H)

        # Query token aggregates information from all patches
        query = self.query_token.expand(B, -1, -1)  # (B, 1, H)
        x_coarse, _ = self.cross_attention(query, x, x)
        x_coarse = self.cross_norm(x_coarse)        # (B, 1, H)

        # Predict coarse-scale outputs
        logits = self.output_proj(x_coarse)         # (B, 1, K)
        latent_pred = self.latent_proj(x_coarse)    # (B, 1, D)

        return logits, latent_pred
