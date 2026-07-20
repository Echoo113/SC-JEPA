import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Decoder Architecture
# ============================================================================
# Patch-level reconstruction decoder for VQ-VAE training
# Reconstructs original patches from latent representations

class Decoder(nn.Module):
    """Enhanced decoder for patch-level reconstruction from quantized latent vectors: (B, N, latent_dim) -> (B, N, C, L)"""
    def __init__(self, latent_dim, out_channels, patch_len):
        super().__init__()
        self.out_channels = out_channels
        self.patch_len = patch_len
        
        # Enhanced decoder for quantized latent vectors (256 dimensions)
        self.net = nn.Sequential(
            # First layer: expand from latent_dim to higher dimension
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Second layer with residual connection
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Third layer
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(128, out_channels * patch_len)
        )
        
        # Residual connection for better gradient flow
        self.residual_proj = nn.Linear(latent_dim, out_channels * patch_len)

    def forward(self, x):
        B, N, D = x.shape  # x is (B, N, latent_dim) where latent_dim = 256
        x_flat = x.view(B * N, D)  # (B*N, latent_dim)
        
        # Main decoder path
        out_flat = self.net(x_flat)  # (B*N, C*L)
        
        # Residual connection for better reconstruction
        residual = self.residual_proj(x_flat)  # (B*N, C*L)
        out_flat = out_flat + residual
        
        out = out_flat.view(B, N, self.out_channels, self.patch_len)  # (B, N, C, L)
        return out
