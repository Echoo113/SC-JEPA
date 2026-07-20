import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# CNN Feature Extractor Architecture
# ============================================================================
# Enhanced CNN with residual connections for robust feature extraction

class ResidualLayer1D(nn.Module):
    """1D residual block with Batch Normalization"""
    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(False),  # fix inplace operation
            nn.BatchNorm1d(in_dim),
            nn.Conv1d(in_dim, res_h_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(False),  # fix inplace operation
            nn.BatchNorm1d(res_h_dim),
            nn.Conv1d(res_h_dim, h_dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # x shape = (B, in_dim, L)
        return x + self.res_block(x)

# ============================================================================
# Enhanced CNN Feature Extractor
# ============================================================================
# Multi-layer CNN with residual blocks and intermediate projections
# Architecture: 2 conv layers + 4 residual layers + intermediate projection

class CnnFeatureExtractor(nn.Module):
    """Lightweight linear feature extractor"""
    def __init__(self, in_channels, h_dim, res_h_dim, n_res_layers, latent_dim, patch_len=20):
        super().__init__()

        # Simple linear projection from patch_len * in_channels to h_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels * patch_len, h_dim),  # Use patch_len parameter
            nn.ReLU(inplace=False),
            nn.Dropout(0.1)
        )

        # Lightweight feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(h_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape = (B, in_channels, patch_len)
        B = x.shape[0]
        # Flatten: (B, in_channels, patch_len) -> (B, in_channels * patch_len)
        x = x.reshape(B, -1)
        x = self.input_proj(x)
        return self.feature_net(x)


# ============================================================================
# Encoder Architecture
# ============================================================================
# Combines CNN feature extractor with Transformer for patch-level representations
# Key: Returns ALL patch representations, not just CLS token
# Architecture: CNN + CLS token + Position embedding + Transformer + Final projection

class Encoder(nn.Module):
    """Encoder = CNN feature extractor + TransformerEncoder. Input: (B, num_patches, patch_len, in_channels), Output: (B, num_patches, latent_dim)"""
    def __init__(self, num_patches, patch_len, latent_dim,
                 cnn_h_dim, cnn_res_h_dim, cnn_n_res_layers,
                 trans_nhead, trans_num_layers, in_channels=1):
        super().__init__()
        self.cnn = CnnFeatureExtractor(
            in_channels, cnn_h_dim, cnn_res_h_dim, cnn_n_res_layers, latent_dim, patch_len
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=trans_nhead,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=trans_num_layers
        )

        self.final_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0,1,3,2).contiguous().view(B*N, C, L)
        feat = self.cnn(x)                          # (B*N, latent_dim)
        feat = feat.view(B, N, -1)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, latent_dim)
        feat = torch.cat([cls_tokens, feat], dim=1)    # (B, N+1, latent_dim)
        feat = feat + self.pos_emb[:, :feat.shape[1], :]

        out = self.transformer(feat)
        all_patch_reprs = out[:, 1:, :]  # (B, N, latent_dim) - exclude CLS token
        return self.final_proj(all_patch_reprs)  # (B, N, latent_dim)
