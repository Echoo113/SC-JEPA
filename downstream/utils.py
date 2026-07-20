# ============================================================================
# Utilities for Downstream Anomaly Prediction
# ============================================================================
# Data loading, model loading, and helper functions

import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

from models.encoder import Encoder
from models.quantizer import Quantizer


# ============================================================================
# Data Loading
# ============================================================================

class PatchDataset(Dataset):
    """
    Dataset for downstream anomaly prediction tasks.
    
    Input shape: (N, P, L, V) where
      N = number of samples
      P = number of patches
      L = patch length (sequence length)
      V = number of variables
    """
    
    def __init__(self, npz_path, patch_len=20):
        data = np.load(npz_path)
        self.x_patches = torch.tensor(data["x_patches"], dtype=torch.float32)
        self.y_label = torch.tensor(data["y_label"], dtype=torch.long)
        self.patch_len = patch_len

        data_patch_len = self.x_patches.shape[2]
        if data_patch_len != patch_len:
            raise ValueError(
                f"Data patch_len ({data_patch_len}) != expected ({patch_len})"
            )

        print(f"📊 Loaded: x_patches {self.x_patches.shape}, y_label {self.y_label.shape}")

        # Convert (N,) to (N,P) if needed - convert patch-level labels to window-level
        if self.y_label.ndim == 1:
            P = self.x_patches.shape[1]
            self.y_label = self.y_label.unsqueeze(1).repeat(1, P)

    def __len__(self):
        return len(self.x_patches)

    def __getitem__(self, idx):
        return self.x_patches[idx], self.y_label[idx]


def load_dataset(npz_path, batch_size=128, shuffle=True, patch_len=20):
    """
    Load dataset and return dataloader with statistics.
    
    Args:
        npz_path: Path to NPZ file
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle data
        patch_len: Expected patch length
    
    Returns:
        loader: DataLoader instance
        dataset: PatchDataset instance
    """
    dataset = PatchDataset(npz_path, patch_len=patch_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    anomaly_ratio = dataset.y_label.float().mean()
    print(f"   Anomaly ratio: {anomaly_ratio:.3f}")
    
    return loader, dataset

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_split(dataset, train_ratio=0.60, val_ratio=0.20, seed=None):
    """
    Split dataset chronologically into train/val/test subsets.

    Default split: 60% train, 20% val, 20% test (time-ordered).

    Args:
        dataset: Full dataset ordered by time
        train_ratio: Fraction for training (default: 0.60)
        val_ratio: Fraction for validation (default: 0.20)
        seed: Unused; kept for backward compatibility (split is time-ordered)

    Returns:
        train_subset, val_subset, test_subset
    """
    total = len(dataset)

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            f"Invalid split sizes (train={train_size}, val={val_size}, test={test_size})."
        )

    indices = list(range(total))
    train_idx = indices[:train_size]
    val_idx = indices[train_size: train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train = torch.utils.data.Subset(dataset, train_idx)
    val = torch.utils.data.Subset(dataset, val_idx)
    test = torch.utils.data.Subset(dataset, test_idx)

    print(f"   Split (time-ordered): train={len(train)}, val={len(val)}, test={len(test)}")

    return train, val, test


# ============================================================================
# Model Loading
# ============================================================================

def _resolve_path(ckpt_name="jepa_vqvae_best.pth"):
    """
    Resolve checkpoint path with fallback options.
    
    Args:
        ckpt_name: Checkpoint filename
    
    Returns:
        Resolved absolute path to checkpoint
        
    Raises:
        FileNotFoundError: If checkpoint not found in any candidate paths
    """
    candidates = [
        ckpt_name,
        os.path.join("./", ckpt_name),
        os.path.join("../", ckpt_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Checkpoint not found: {candidates}")


def create_encoder(num_patches=5, patch_len=20, latent_dim=256,
                   cnn_h_dim=64, cnn_res_h_dim=32, cnn_n_res_layers=2,
                   trans_nhead=8, trans_num_layers=6, in_channels=1):
    """
    Create encoder model with specified architecture.
    
    Args:
        num_patches: Number of patches (P)
        patch_len: Patch length (sequence length)
        latent_dim: Encoder latent dimension (D)
        cnn_h_dim: CNN hidden dimension
        cnn_res_h_dim: CNN residual block hidden dimension
        cnn_n_res_layers: Number of CNN residual layers
        trans_nhead: Transformer attention heads
        trans_num_layers: Transformer encoder layers
        in_channels: Input channels (default: 1 for univariate)
    
    Returns:
        Encoder instance
    """
    return Encoder(
        num_patches=num_patches,
        patch_len=patch_len,
        latent_dim=latent_dim,
        cnn_h_dim=cnn_h_dim,
        cnn_res_h_dim=cnn_res_h_dim,
        cnn_n_res_layers=cnn_n_res_layers,
        trans_nhead=trans_nhead,
        trans_num_layers=trans_num_layers,
        in_channels=in_channels
    )


def create_quantizer(num_codes=128, embedding_dim=256, temperature=0.1):
    """
    Create quantizer model.
    
    Args:
        num_codes: Number of codebook entries (K)
        embedding_dim: Embedding dimension (D)
        temperature: Softmax temperature for soft assignment
    
    Returns:
        Quantizer instance
    """
    return Quantizer(
        num_codes=num_codes,
        embedding_dim=embedding_dim,
        temperature=temperature
    )


def load_pretrained(encoder, quantizer, device, ckpt_path=None, temp=0.65):
    """
    Load pretrained weights and freeze models for inference.
    
    Args:
        encoder: Encoder instance
        quantizer: Quantizer instance
        device: torch device (cpu or cuda)
        ckpt_path: Checkpoint path (auto-resolve if None)
        temp: Quantizer temperature to set
    
    Returns:
        encoder: Frozen encoder on device
        quantizer: Frozen quantizer on device
    """
    path = _resolve_path(ckpt_path or "jepa_vqvae_best.pth")
    ckpt = torch.load(path, map_location=device)

    # Override temperature if in checkpoint
    if "quantizer_temperature" in ckpt:
        temp = ckpt["quantizer_temperature"]

    encoder.load_state_dict(ckpt["encoder_online"])
    quantizer.load_state_dict(ckpt["quantizer_online"])
    quantizer.temperature = temp

    # Move to device and freeze
    encoder = encoder.to(device)
    quantizer = quantizer.to(device)
    
    for m in [encoder, quantizer]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    print(f"✅ Loaded: {path}")
    print(f"   Quantizer T={quantizer.temperature:.2f}")
    
    return encoder, quantizer
